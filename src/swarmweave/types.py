"""Public pydantic types used across swarmweave.

These models form the contract between the Supervisor, Worker, and Backend
layers. Everything that flows through the swarm is one of these types.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> float:
    return time.time()


class Observation(BaseModel):
    """A single contribution to the shared context log.

    An Observation is the unit of memory. Every meaningful action a worker
    takes — a search result, a draft, a finding — is recorded as one of
    these so other workers and the supervisor can build on it.
    """

    id: str = Field(default_factory=_uuid)
    agent_id: str
    task: str
    content: str
    kind: Literal["finding", "draft", "tool_result", "note"] = "finding"
    timestamp: float = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextSlice(BaseModel):
    """The outcome-optimized view of shared context returned to a worker.

    A slice is a relevance- and recency-ranked subset of all observations
    in the swarm's shared log, scoped to what this specific worker needs
    to do its current subtask well.
    """

    observations: list[Observation] = Field(default_factory=list)
    summary: str | None = None
    tokens_estimate: int = 0

    def render(self) -> str:
        """Render the slice as a flat string suitable for an LLM prompt."""
        if not self.observations:
            return "(no prior shared context yet)"
        parts: list[str] = []
        if self.summary:
            parts.append(f"Summary of prior work:\n{self.summary}\n")
        parts.append("Relevant prior observations from other agents:")
        for obs in self.observations:
            parts.append(f"- [{obs.agent_id} / {obs.kind}] {obs.content.strip()}")
        return "\n".join(parts)


class Outcome(BaseModel):
    """End-of-run signal used to close the credit-assignment loop."""

    success: bool
    score: float | None = None
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkerSpec(BaseModel):
    """Static configuration for a worker.

    A WorkerSpec is what users hand to the Supervisor. It is intentionally
    lightweight — the heavy machinery lives in :class:`swarmweave.Worker`.
    """

    name: str
    role: str
    model: str = "gpt-4o-mini"
    system_prompt: str | None = None
    tool_names: list[str] = Field(default_factory=list)


class Task(BaseModel):
    """A unit of work assigned to a single worker."""

    id: str = Field(default_factory=_uuid)
    description: str
    parent_id: str | None = None
    assigned_to: str


class SwarmMetrics(BaseModel):
    """Aggregate metrics for one Supervisor.run invocation."""

    input_tokens: int = 0
    output_tokens: int = 0
    wall_clock_seconds: float = 0.0
    worker_calls: int = 0


class SwarmResult(BaseModel):
    """The structured return value of :meth:`Supervisor.run`."""

    final_output: str
    per_worker_outputs: dict[str, str] = Field(default_factory=dict)
    trace: list[Observation] = Field(default_factory=list)
    metrics: SwarmMetrics = Field(default_factory=SwarmMetrics)
