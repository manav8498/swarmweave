"""User-facing wrapper around a :class:`Backend`.

:class:`SharedContext` is the centerpiece of swarmweave's public API. It
provides both async primitives (used internally by the Supervisor and Worker
machinery) and ergonomic sync wrappers for users who want to inspect or
manipulate context from a notebook or script.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any, TypeVar

from swarmweave.backends.base import Backend
from swarmweave.backends.local import LocalBackend
from swarmweave.events import EventBus
from swarmweave.types import ContextSlice, Observation, Outcome

logger = logging.getLogger("swarmweave")

T = TypeVar("T")


def _run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run *coro* to completion, raising a clear error if a loop is already running."""
    try:
        loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        # Avoid the "coroutine was never awaited" warning before raising.
        coro.close()
        raise RuntimeError(
            "Cannot call the sync API from inside a running event loop. "
            "Use the async variant (e.g. await ctx.aread(...)) or wrap your "
            "call site with asyncio.run()."
        )
    return asyncio.run(coro)


class SharedContext:
    """Outcome-optimized shared memory for a swarm.

    Parameters
    ----------
    backend:
        The backend that owns persistence and retrieval. Defaults to a
        :class:`LocalBackend` with no session id, so every fresh
        ``SharedContext()`` starts a new in-memory log.

    Examples
    --------
    >>> from swarmweave import SharedContext
    >>> ctx = SharedContext()
    >>> # supervisors and workers will read/write through this object
    """

    def __init__(self, backend: Backend | None = None, events: EventBus | None = None) -> None:
        self.backend: Backend = backend if backend is not None else LocalBackend()
        self.events: EventBus | None = events

    # ---- async primitives (used by Supervisor/Worker) ----

    async def aread(self, agent_id: str, task: str, limit: int = 8) -> ContextSlice:
        """Async: return the slice of context most relevant to ``task``."""
        slice_ = await self.backend.query(agent_id=agent_id, task=task, limit=limit)
        if self.events is not None:
            self.events.emit(
                "context_read",
                actor=agent_id,
                task=task,
                hit_count=len(slice_.observations),
                hit_ids=[o.id for o in slice_.observations],
            )
        return slice_

    async def awrite(self, agent_id: str, observation: Observation) -> None:
        """Async: append an observation contributed by ``agent_id``."""
        if observation.agent_id != agent_id:
            observation = observation.model_copy(update={"agent_id": agent_id})
        await self.backend.append(observation)
        if self.events is not None:
            self.events.emit(
                "context_write",
                actor=agent_id,
                observation_id=observation.id,
                content_preview=observation.content[:200],
                obs_kind=observation.kind,
            )

    async def afinalize(self, outcome: Outcome) -> None:
        """Async: record a final outcome for the run."""
        await self.backend.finalize(outcome)

    async def aget_full_trace(self) -> list[Observation]:
        """Async: return every observation in the log."""
        return await self.backend.trace()

    # ---- sync wrappers for users ----

    def read(self, agent_id: str, task: str, limit: int = 8) -> ContextSlice:
        """Return the slice of context most relevant to ``task``."""
        return _run_sync(self.aread(agent_id=agent_id, task=task, limit=limit))

    def write(self, agent_id: str, observation: Observation) -> None:
        """Append an observation contributed by ``agent_id``."""
        _run_sync(self.awrite(agent_id=agent_id, observation=observation))

    def finalize(self, outcome: Outcome) -> None:
        """Record a final outcome for the run."""
        _run_sync(self.afinalize(outcome))

    def get_full_trace(self) -> list[Observation]:
        """Return every observation in the log."""
        result = _run_sync(self.aget_full_trace())
        return list(result)
