"""The Worker — one specialized agent inside a swarm.

A Worker holds its static configuration (name, role, model, tools, optional
system prompt) and exposes one execution primitive, :meth:`arun`, which:

1. asks the :class:`SharedContext` for the slice of prior swarm work that
   is most relevant to *this* worker's subtask,
2. lets the model autonomously pick tools and produce a final answer,
3. writes the result back into the shared log so downstream workers can
   build on it.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

from swarmweave._openai_client import OpenAIClient
from swarmweave._tools import build_tool_registry
from swarmweave.events import EventBus
from swarmweave.shared_context import SharedContext
from swarmweave.types import Observation

logger = logging.getLogger("swarmweave")

_DEFAULT_SYSTEM = """You are {name}, a specialist agent in a coordinated swarm.

Your role: {role}

You are one of several agents working on a shared task. Other agents have
likely produced findings already; treat them as trusted teammates and build
on their work rather than repeating it. When you call a tool, do it because
it advances your subtask, not because the tool exists.

Produce a single, concrete deliverable for your subtask. Be specific. Cite
the prior observations you used by quoting a short phrase from them when
relevant."""


class Worker:
    """Configuration + execution for one specialist agent.

    Parameters
    ----------
    name:
        Stable identifier used as the agent id in the shared log.
    role:
        Short, human-readable description of what this worker does. Used
        both in the system prompt and by the supervisor when deciding which
        subtask to route here.
    model:
        OpenAI model identifier. Defaults to ``gpt-4o-mini``.
    system_prompt:
        Override the default system prompt entirely. ``{name}`` and
        ``{role}`` placeholders are interpolated if present.
    tools:
        Optional list of Python callables exposed to the model. Each
        callable must have a docstring and typed parameters; see
        :func:`swarmweave._tools.callable_to_tool_spec`.
    max_tool_iterations:
        Safety bound on the tool-use loop. Defaults to 6.
    """

    def __init__(
        self,
        name: str,
        role: str,
        model: str = "gpt-4o-mini",
        system_prompt: str | None = None,
        tools: list[Callable[..., Any]] | None = None,
        max_tool_iterations: int = 6,
    ) -> None:
        if not name or not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"Worker name must be a non-empty alphanumeric string (with - or _); got {name!r}."
            )
        self.name = name
        self.role = role
        self.model = model
        self.system_prompt = system_prompt
        self.tools = list(tools or [])
        self.max_tool_iterations = max_tool_iterations
        self._tool_specs, self._tool_registry = build_tool_registry(self.tools)

    def _render_system(self) -> str:
        template = self.system_prompt or _DEFAULT_SYSTEM
        try:
            return template.format(name=self.name, role=self.role)
        except (KeyError, IndexError):
            return template

    def _build_user_message(self, subtask: str, context_block: str) -> str:
        return (
            f"Your subtask:\n{subtask}\n\n"
            f"Shared context from the rest of the swarm:\n{context_block}\n\n"
            "Produce your deliverable now. End with a single concise paragraph "
            "labeled 'FINDING:' that another agent could quote directly."
        )

    async def _invoke_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        func = self._tool_registry.get(tool_name)
        if func is None:
            return f"error: tool {tool_name!r} is not registered for this worker."
        try:
            result = func(**tool_input)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            logger.exception("tool %s raised", tool_name)
            return f"error: {type(exc).__name__}: {exc}"
        return str(result)

    async def arun(
        self,
        subtask: str,
        shared_context: SharedContext,
        client: OpenAIClient,
        events: EventBus | None = None,
    ) -> Observation:
        """Run the worker on ``subtask`` and return its final observation."""
        import time as _time

        start = _time.perf_counter()
        logger.info("worker %s starting: %s", self.name, subtask[:120])
        if events is not None:
            events.emit("worker_start", actor=self.name, subtask=subtask, role=self.role)
        slice_ = await shared_context.aread(agent_id=self.name, task=subtask)
        context_block = slice_.render()

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": self._build_user_message(subtask, context_block)}
        ]

        final_text: str = ""
        exhausted = True
        for _iteration in range(self.max_tool_iterations + 1):
            result = await client.call(
                system=self._render_system(),
                messages=messages,
                tools=self._tool_specs or None,
                model=self.model,
            )
            if result.finish_reason != "tool_calls" or not result.tool_calls:
                final_text = result.text
                exhausted = False
                break

            messages.append(result.raw_assistant_message)
            for call in result.tool_calls:
                if events is not None:
                    events.emit(
                        "worker_tool_call",
                        actor=self.name,
                        tool=call.name,
                        arguments=call.arguments,
                    )
                output = await self._invoke_tool(call.name, call.arguments)
                if events is not None:
                    events.emit(
                        "worker_tool_result",
                        actor=self.name,
                        tool=call.name,
                        output_preview=output[:200],
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": output,
                    }
                )

        if exhausted:
            logger.warning(
                "worker %s hit max_tool_iterations=%d; forcing wrap-up call",
                self.name,
                self.max_tool_iterations,
            )
            if events is not None:
                events.emit("worker_wrap_up", actor=self.name)
            # Forced wrap-up: no tools available, commit to a FINDING based on
            # whatever evidence the model already gathered. Without this, reasoning
            # models tend to keep exploring forever and never produce a deliverable.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have used your tool budget. Do not request more tools. "
                        "Based on the evidence you have already gathered above, "
                        "produce your final deliverable now and end with a single "
                        "concise paragraph labeled 'FINDING:'."
                    ),
                }
            )
            wrap_up = await client.call(
                system=self._render_system(),
                messages=messages,
                tools=None,
                model=self.model,
            )
            final_text = wrap_up.text or final_text or "(no final answer — wrap-up empty)"

        observation = Observation(
            agent_id=self.name,
            task=subtask,
            content=final_text.strip() or "(empty response)",
            kind="finding",
            metadata={"role": self.role, "model": self.model},
        )
        await shared_context.awrite(self.name, observation)
        elapsed = _time.perf_counter() - start
        logger.info("worker %s done in %.1fs", self.name, elapsed)
        if events is not None:
            events.emit(
                "worker_done",
                actor=self.name,
                wall_clock=round(elapsed, 2),
                content_preview=observation.content[:300],
            )
        return observation

    def run(
        self,
        subtask: str,
        shared_context: SharedContext,
        client: OpenAIClient,
    ) -> Observation:
        """Synchronous wrapper around :meth:`arun`."""
        import asyncio

        return asyncio.run(self.arun(subtask, shared_context, client))
