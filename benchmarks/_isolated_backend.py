"""IsolatedBackend — used to simulate a no-shared-memory baseline.

Workers can write into this backend, but every read returns an empty slice.
This mirrors the dominant pattern across multi-agent frameworks today:
each subagent runs with its own private context window and the supervisor
stitches outputs together at the end.

Used only by ``baseline_isolated.py`` so that both benchmark conditions
share the *exact* same Supervisor / Worker / tool plumbing — only the
context layer differs.
"""

from __future__ import annotations

import asyncio

from swarmweave.backends.base import Backend
from swarmweave.types import ContextSlice, Observation, Outcome


class IsolatedBackend(Backend):
    """Append-only log whose query() always returns an empty slice."""

    def __init__(self) -> None:
        self._log: list[Observation] = []
        self._outcome: Outcome | None = None
        self._lock = asyncio.Lock()

    async def append(self, observation: Observation) -> None:
        async with self._lock:
            self._log.append(observation)

    async def query(self, agent_id: str, task: str, limit: int = 8) -> ContextSlice:
        # Isolated baseline: workers cannot see each other's work.
        return ContextSlice()

    async def trace(self) -> list[Observation]:
        async with self._lock:
            return list(self._log)

    async def finalize(self, outcome: Outcome) -> None:
        async with self._lock:
            self._outcome = outcome
