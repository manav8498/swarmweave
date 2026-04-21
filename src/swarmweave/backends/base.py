"""Abstract Backend interface for the shared-context layer.

A Backend owns persistence and retrieval. The :class:`SharedContext` façade
delegates to whatever Backend it was constructed with, which lets users swap
the local file-based implementation for a remote service without touching
their swarm code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from swarmweave.types import ContextSlice, Observation, Outcome


class Backend(ABC):
    """Interface every shared-context backend must implement.

    Implementations are expected to be safe to call from multiple coroutines
    concurrently; the :class:`LocalBackend` reference implementation uses an
    asyncio lock around its mutating methods.
    """

    @abstractmethod
    async def append(self, observation: Observation) -> None:
        """Persist a new observation contributed by some agent."""

    @abstractmethod
    async def query(self, agent_id: str, task: str, limit: int = 8) -> ContextSlice:
        """Return the slice of the shared log most relevant to ``task``.

        Implementations should return what ``agent_id`` needs to know *now*,
        given everything the swarm has already done. This is where outcome-
        aware retrieval would plug in — for the local reference backend it is
        a simple text-similarity + recency score.
        """

    @abstractmethod
    async def trace(self) -> list[Observation]:
        """Return the full append-only log for this session."""

    @abstractmethod
    async def finalize(self, outcome: Outcome) -> None:
        """Record a final outcome for credit-assignment / analytics."""

    async def close(self) -> None:
        """Release any held resources. Default is a no-op."""
        return None
