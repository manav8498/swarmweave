"""swarmweave — supervisor-orchestrated multi-agent swarms with shared context.

Public API:

* :class:`SharedContext` — outcome-optimized shared memory for a swarm.
* :class:`Supervisor` — task decomposition + parallel orchestration.
* :class:`Worker` — one specialist agent.
* :class:`Observation`, :class:`Outcome` — types you'll see in results.
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

from swarmweave.events import EventBus, SwarmEvent
from swarmweave.lessons import LessonBook, Mentor
from swarmweave.shared_context import SharedContext
from swarmweave.supervisor import Supervisor
from swarmweave.types import (
    ContextSlice,
    Observation,
    Outcome,
    SwarmMetrics,
    SwarmResult,
    Task,
    WorkerSpec,
)
from swarmweave.worker import Worker

try:
    __version__ = version("swarmweave")
except PackageNotFoundError:  # editable install during early dev
    __version__ = "0.0.0+unknown"

# Library uses a module-level logger; library code never calls print().
logging.getLogger("swarmweave").addHandler(logging.NullHandler())

__all__ = [
    "ContextSlice",
    "EventBus",
    "LessonBook",
    "Mentor",
    "Observation",
    "Outcome",
    "SharedContext",
    "Supervisor",
    "SwarmEvent",
    "SwarmMetrics",
    "SwarmResult",
    "Task",
    "Worker",
    "WorkerSpec",
    "__version__",
]
