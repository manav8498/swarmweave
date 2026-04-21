"""The Supervisor — task decomposition and parallel orchestration.

The Supervisor is the top-level orchestrator. It accepts a user task and a
list of :class:`Worker` instances, decomposes the task using a single OpenAI
call, fans the resulting subtasks out across workers in parallel via the
LangGraph adapter, and finally synthesizes a structured result.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Literal

from swarmweave._openai_client import OpenAIClient
from swarmweave.events import EventBus
from swarmweave.shared_context import SharedContext
from swarmweave.types import Observation, Outcome, SwarmMetrics, SwarmResult, Task
from swarmweave.worker import Worker

SwarmMode = Literal["parallel", "sequential"]

logger = logging.getLogger("swarmweave")


_DECOMPOSE_SYSTEM = """You are the supervisor of a small expert swarm.

You are given a user task and the roster of available workers (each with a
name and a role). Your job is to assign exactly one subtask to each worker.

Subtasks should:
- be self-contained enough that the worker can act on them in isolation,
- be aligned with the worker's role,
- together cover the user's task without significant overlap.

Use the assign_subtasks tool. Do not respond in plain text."""


_SYNTHESIZE_SYSTEM = """You are the supervisor of a small expert swarm.

You are given the user's original task and the deliverables produced by each
worker. Your job is to produce one cohesive final answer for the user.

- Cite each worker by name when integrating their findings.
- Resolve contradictions explicitly rather than picking one silently.
- If a worker reported a problem, surface it.
- Do not pad. Length should match what the task actually requires."""


_ASSIGN_TOOL = {
    "type": "function",
    "function": {
        "name": "assign_subtasks",
        "description": "Assign exactly one subtask per worker.",
        "parameters": {
            "type": "object",
            "properties": {
                "assignments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "worker": {
                                "type": "string",
                                "description": "Name of the worker to assign this subtask to.",
                            },
                            "subtask": {
                                "type": "string",
                                "description": "What the worker should do, in one or two sentences.",
                            },
                        },
                        "required": ["worker", "subtask"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["assignments"],
            "additionalProperties": False,
        },
    },
}


class Supervisor:
    """Orchestrates a swarm of :class:`Worker` instances over a shared context.

    Parameters
    ----------
    workers:
        The roster of workers available to this swarm.
    shared_context:
        Shared memory the swarm reads from and writes to. Defaults to a
        fresh in-memory :class:`SharedContext`.
    model:
        Model used for decomposition and synthesis. Defaults to ``gpt-4o-mini``.
    client:
        Optional pre-built :class:`OpenAIClient`. If omitted, one is created
        from environment credentials on first use.
    mode:
        ``"parallel"`` (default): all workers fan out simultaneously via
        LangGraph's ``Send`` API. Best when workers cover non-overlapping
        angles and do not depend on each other.
        ``"sequential"``: workers run one at a time in the order they appear
        in ``workers``. Each worker's :meth:`SharedContext.aread` returns
        every prior worker's findings, so this is the right mode for true
        pipelines (classify → draft → verify, etc.).
    """

    def __init__(
        self,
        workers: list[Worker],
        shared_context: SharedContext | None = None,
        model: str = "gpt-4o-mini",
        client: OpenAIClient | None = None,
        mode: SwarmMode = "parallel",
        events: EventBus | None = None,
    ) -> None:
        if not workers:
            raise ValueError("Supervisor requires at least one worker.")
        names = [w.name for w in workers]
        if len(set(names)) != len(names):
            raise ValueError(f"Worker names must be unique; got {names!r}.")
        if mode not in ("parallel", "sequential"):
            raise ValueError(f"mode must be 'parallel' or 'sequential'; got {mode!r}.")
        self.workers = workers
        self.shared_context = shared_context if shared_context is not None else SharedContext()
        self.model = model
        self._client = client
        self.mode: SwarmMode = mode
        self.events: EventBus | None = events
        # Forward events to the shared context so context_read/write fire too.
        if events is not None and self.shared_context.events is None:
            self.shared_context.events = events

    @property
    def client(self) -> OpenAIClient:
        if self._client is None:
            self._client = OpenAIClient(model=self.model)
        return self._client

    def _worker_by_name(self, name: str) -> Worker | None:
        for w in self.workers:
            if w.name == name:
                return w
        return None

    async def adecompose(self, user_task: str) -> list[Task]:
        """Use the model to assign exactly one subtask per worker."""
        logger.info("decomposing task across %d workers", len(self.workers))
        if self.events is not None:
            self.events.emit(
                "decompose_start",
                actor="supervisor",
                user_task=user_task,
                worker_names=[w.name for w in self.workers],
            )
        roster = "\n".join(f"- {w.name}: {w.role}" for w in self.workers)
        user_msg = (
            f"User task:\n{user_task}\n\n"
            f"Workers available ({len(self.workers)}):\n{roster}\n\n"
            "Call assign_subtasks now with one assignment per worker."
        )
        result = await self.client.call(
            system=_DECOMPOSE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            tools=[_ASSIGN_TOOL],
            tool_choice={"type": "function", "function": {"name": "assign_subtasks"}},
            model=self.model,
        )

        assignments: list[dict[str, Any]] = []
        for tc in result.tool_calls:
            if tc.name == "assign_subtasks":
                payload = tc.arguments or {}
                assignments = list(payload.get("assignments", []))
                break

        if not assignments:
            # Fallback: assign each worker its role verbatim.
            logger.warning(
                "decompose returned no tool call; falling back to role-based assignment."
            )
            assignments = [
                {"worker": w.name, "subtask": f"{w.role} for: {user_task}"} for w in self.workers
            ]

        # Ensure every named worker gets exactly one subtask.
        seen: set[str] = set()
        tasks: list[Task] = []
        for entry in assignments:
            worker_name = entry.get("worker", "").strip()
            subtask = entry.get("subtask", "").strip()
            if not worker_name or worker_name in seen:
                continue
            if self._worker_by_name(worker_name) is None:
                logger.warning("decompose referenced unknown worker %r; skipping.", worker_name)
                continue
            seen.add(worker_name)
            tasks.append(Task(description=subtask, assigned_to=worker_name))

        for w in self.workers:
            if w.name not in seen:
                tasks.append(Task(description=f"{w.role} for: {user_task}", assigned_to=w.name))

        if self.events is not None:
            self.events.emit(
                "decompose_done",
                actor="supervisor",
                assignments=[{"worker": t.assigned_to, "subtask": t.description} for t in tasks],
            )
        return tasks

    async def asynthesize(
        self,
        user_task: str,
        worker_outputs: dict[str, str],
    ) -> str:
        """Produce the final answer from per-worker outputs."""
        logger.info("synthesizing final answer from %d worker outputs", len(worker_outputs))
        if self.events is not None:
            self.events.emit(
                "synthesize_start",
                actor="supervisor",
                worker_count=len(worker_outputs),
            )
        block = "\n\n".join(f"### {name}\n{output}" for name, output in worker_outputs.items())
        user_msg = (
            f"User task:\n{user_task}\n\n"
            f"Worker deliverables:\n{block}\n\n"
            "Write the final answer for the user now."
        )
        result = await self.client.call(
            system=_SYNTHESIZE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            model=self.model,
        )
        text = result.text.strip() or "(empty synthesis)"
        if self.events is not None:
            self.events.emit(
                "synthesize_done",
                actor="supervisor",
                final_output_preview=text[:300],
            )
        return text

    async def arun(self, user_task: str) -> SwarmResult:
        """Run the full decompose → fan-out → synthesize → finalize pipeline."""
        from swarmweave.adapters.langgraph_adapter import build_swarm_graph

        if not user_task or not user_task.strip():
            raise ValueError("user_task must be a non-empty string.")

        if self.events is not None:
            self.events.emit(
                "run_start",
                actor="supervisor",
                user_task=user_task,
                workers=[{"name": w.name, "role": w.role} for w in self.workers],
                mode=self.mode,
                model=self.model,
            )

        start = time.perf_counter()
        graph = build_swarm_graph(self)
        initial_state: dict[str, Any] = {
            "user_task": user_task,
            "subtasks_json": "",
            "worker_outputs": {},
            "final_output": "",
        }
        final_state: dict[str, Any] = await graph.ainvoke(initial_state)
        wall_clock = time.perf_counter() - start

        trace = await self.shared_context.aget_full_trace()
        worker_outputs = dict(final_state.get("worker_outputs", {}))
        metrics = SwarmMetrics(
            input_tokens=self.client.stats.input_tokens,
            output_tokens=self.client.stats.output_tokens,
            wall_clock_seconds=round(wall_clock, 3),
            worker_calls=self.client.stats.calls,
        )

        await self.shared_context.afinalize(
            Outcome(success=True, notes="run completed", metadata={"task": user_task})
        )

        # Proactively release backend resources (e.g. the embedding-API http
        # pool) so they don't get garbage-collected after the event loop is
        # torn down, which triggers a spurious "no running event loop" warning.
        import contextlib

        with contextlib.suppress(Exception):
            await self.shared_context.backend.close()

        result = SwarmResult(
            final_output=final_state.get("final_output", ""),
            per_worker_outputs=worker_outputs,
            trace=list(trace),
            metrics=metrics,
        )
        if self.events is not None:
            self.events.emit(
                "run_done",
                actor="supervisor",
                wall_clock=metrics.wall_clock_seconds,
                worker_calls=metrics.worker_calls,
                input_tokens=metrics.input_tokens,
                output_tokens=metrics.output_tokens,
            )
            await self.events.aclose()
        return result

    def run(self, user_task: str) -> SwarmResult:
        """Synchronous wrapper around :meth:`arun`."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(user_task))
        raise RuntimeError(
            "Supervisor.run was called from inside a running event loop. "
            "Use `await supervisor.arun(...)` instead."
        )

    # Helpers used by the LangGraph adapter ----------------------------------------

    async def _run_one_worker(self, worker_name: str, subtask: str) -> Observation:
        worker = self._worker_by_name(worker_name)
        if worker is None:
            raise KeyError(f"Worker {worker_name!r} not found in roster.")
        return await worker.arun(subtask, self.shared_context, self.client, events=self.events)

    @staticmethod
    def _serialize_tasks(tasks: list[Task]) -> str:
        return json.dumps([t.model_dump() for t in tasks])

    @staticmethod
    def _deserialize_tasks(blob: str) -> list[Task]:
        if not blob:
            return []
        return [Task.model_validate(item) for item in json.loads(blob)]
