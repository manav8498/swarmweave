"""Live event stream for swarm runs.

The :class:`EventBus` is an opt-in observer that the Supervisor, Worker, and
SharedContext can publish to. Consumers (the live TUI, custom dashboards,
your own logging pipeline) subscribe and receive every observation, tool
call, and worker state transition as it happens — without changing any
public API of the swarm itself.

If you don't pass an EventBus, nothing is emitted and there is zero overhead.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

EventKind = Literal[
    "run_start",
    "decompose_start",
    "decompose_done",
    "worker_start",
    "worker_thought",
    "worker_tool_call",
    "worker_tool_result",
    "worker_wrap_up",
    "worker_done",
    "synthesize_start",
    "synthesize_done",
    "context_write",
    "context_read",
    "tokens_update",
    "run_done",
]


class SwarmEvent:
    """One thing that happened during a swarm run."""

    __slots__ = ("actor", "kind", "payload", "timestamp")

    def __init__(self, kind: EventKind, actor: str, payload: dict[str, Any] | None = None) -> None:
        self.timestamp = time.time()
        self.kind: EventKind = kind
        self.actor = actor
        self.payload: dict[str, Any] = payload or {}

    def __repr__(self) -> str:
        return f"SwarmEvent(kind={self.kind!r}, actor={self.actor!r})"


class EventBus:
    """A bounded async queue of :class:`SwarmEvent`s.

    Producers call :meth:`emit`. Consumers ``async for event in bus``.

    The bus is bounded; if a slow consumer would block a producer, the
    producer drops the event with a counter increment instead of blocking
    the swarm. This keeps the swarm fast even when nothing is consuming.
    """

    def __init__(self, maxsize: int = 1024) -> None:
        self._queue: asyncio.Queue[SwarmEvent | None] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0
        self._closed = False

    def emit(self, kind: EventKind, actor: str, **payload: Any) -> None:
        if self._closed:
            return
        event = SwarmEvent(kind=kind, actor=actor, payload=payload)
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped += 1

    async def aclose(self) -> None:
        self._closed = True
        await self._queue.put(None)  # poison pill for consumers

    def __aiter__(self) -> EventBus:
        return self

    async def __anext__(self) -> SwarmEvent:
        event = await self._queue.get()
        if event is None:
            raise StopAsyncIteration
        return event
