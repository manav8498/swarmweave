# Custom backends

A backend owns persistence and retrieval for a `SharedContext`. Swapping in your own backend is the supported way to plug swarmweave into your existing memory stack — Postgres + pgvector, Redis, an internal microservice, or anything else that fits the four-method interface.

## The interface

```python
from abc import ABC, abstractmethod
from swarmweave.types import ContextSlice, Observation, Outcome


class Backend(ABC):
    @abstractmethod
    async def append(self, observation: Observation) -> None:
        """Persist a new observation contributed by some agent."""

    @abstractmethod
    async def query(
        self, agent_id: str, task: str, limit: int = 8
    ) -> ContextSlice:
        """Return the slice of the shared log most relevant to ``task``."""

    @abstractmethod
    async def trace(self) -> list[Observation]:
        """Return the full append-only log for this session."""

    @abstractmethod
    async def finalize(self, outcome: Outcome) -> None:
        """Record a final outcome for credit-assignment / analytics."""

    async def close(self) -> None:
        """Optional. Release any held resources. Default is a no-op."""
```

That's the whole substrate. Everything else — the supervisor, workers, LangGraph adapter — is built on top of these four methods.

## A worked example: Postgres + pgvector

```python
import asyncio
from typing import Any

import asyncpg

from swarmweave.backends.base import Backend
from swarmweave.types import ContextSlice, Observation, Outcome


class PgVectorBackend(Backend):
    def __init__(self, dsn: str, session_id: str, embed: Any):
        self.dsn = dsn
        self.session_id = session_id
        self.embed = embed
        self._pool: asyncpg.Pool | None = None

    async def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.dsn)
        return self._pool

    async def append(self, observation: Observation) -> None:
        pool = await self._ensure_pool()
        vec = await asyncio.to_thread(self.embed, observation.content)
        async with pool.acquire() as conn:
            await conn.execute(
                "insert into observations (session_id, agent_id, task, content, embedding, ts) "
                "values ($1, $2, $3, $4, $5, to_timestamp($6))",
                self.session_id,
                observation.agent_id,
                observation.task,
                observation.content,
                vec,
                observation.timestamp,
            )

    async def query(
        self, agent_id: str, task: str, limit: int = 8
    ) -> ContextSlice:
        pool = await self._ensure_pool()
        vec = await asyncio.to_thread(self.embed, task)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "select agent_id, task, content, ts from observations "
                "where session_id = $1 and agent_id <> $2 "
                "order by embedding <-> $3 limit $4",
                self.session_id,
                agent_id,
                vec,
                limit,
            )
        observations = [
            Observation(
                agent_id=r["agent_id"],
                task=r["task"],
                content=r["content"],
                timestamp=r["ts"].timestamp(),
            )
            for r in rows
        ]
        return ContextSlice(observations=observations)

    async def trace(self) -> list[Observation]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "select agent_id, task, content, ts from observations "
                "where session_id = $1 order by ts asc",
                self.session_id,
            )
        return [
            Observation(
                agent_id=r["agent_id"],
                task=r["task"],
                content=r["content"],
                timestamp=r["ts"].timestamp(),
            )
            for r in rows
        ]

    async def finalize(self, outcome: Outcome) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "insert into outcomes (session_id, success, score, notes) "
                "values ($1, $2, $3, $4)",
                self.session_id,
                outcome.success,
                outcome.score,
                outcome.notes,
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
```

You then construct `SharedContext(backend=PgVectorBackend(...))` and the rest of swarmweave uses it transparently.

## Design checklist

When implementing your own backend, the things actually worth getting right:

1. **Concurrency.** `append()` and `query()` may be called concurrently from parallel workers. Hold a lock or use a connection pool — don't trust order-of-arrival.
2. **`agent_id` filtering.** Most retrieval implementations should *exclude* observations written by the requesting `agent_id` from the slice. Echoing a worker's own writes back to it is rarely useful and inflates token cost.
3. **Bounded slice size.** Respect `limit`. The supervisor and workers send sane defaults (8). Returning everything will blow context windows on long-running swarms.
4. **Cheap `trace()`.** It's used at end-of-run for benchmarking and debugging. Don't recompute embeddings or rerank — return the raw log in time order.
5. **Idempotent `finalize()`.** Treat it as best-effort signal recording. Don't make the swarm fail if your analytics store is briefly unavailable; log and move on.

