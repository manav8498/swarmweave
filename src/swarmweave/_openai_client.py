"""Thin wrapper around the OpenAI Chat Completions API.

Centralizing the call here lets us:

* gracefully degrade when the installed SDK or model rejects newer knobs
  (``reasoning_effort``, ``parallel_tool_calls``);
* swap in a fake during tests without monkey-patching individual call sites;
* keep token-accounting in one place.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("swarmweave")

DEFAULT_MODEL = "gpt-4o-mini"
# Conservative default — gpt-4o-mini caps at 16384, gpt-4o at 16384, o1/o3
# allow much more. The graceful fallback in _call_with_fallback clamps
# down on any "max_tokens too large" error from the API.
DEFAULT_MAX_TOKENS = 8_000


@dataclass
class ToolCall:
    """Normalized tool-call surfaced to callers."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class CallResult:
    """The pieces of an OpenAI response we actually use downstream."""

    text: str
    tool_calls: list[ToolCall]
    finish_reason: str
    raw_assistant_message: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ClientStats:
    """Cumulative token usage across every call made through one client."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    per_model: dict[str, dict[str, int]] = field(default_factory=dict)


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return result if isinstance(result, dict) else {}


class OpenAIClient:
    """Thin async-friendly wrapper around ``openai.AsyncOpenAI``.

    Parameters
    ----------
    api_key:
        Optional override. Falls back to ``OPENAI_API_KEY``.
    model:
        Default model used when a per-call ``model`` is not given.
    max_tokens:
        Default max output tokens per call (forwarded as
        ``max_completion_tokens``).
    reasoning_effort:
        Reasoning effort hint forwarded to the model. Set to ``None`` to
        omit the field entirely. Models that do not support it are handled
        by graceful retry.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning_effort: str | None = "high",
    ) -> None:
        from openai import AsyncOpenAI  # local import keeps import-time light

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "Set OPENAI_API_KEY in your environment or .env file; see .env.example."
            )
        self._client = AsyncOpenAI(api_key=resolved_key)
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.stats = ClientStats()
        self._unsupported_kwargs: set[str] = set()

    async def call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        parallel_tool_calls: bool = True,
    ) -> CallResult:
        """Make one Chat Completions API call and return the parsed result."""
        chosen_model = model or self.model
        full_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            *messages,
        ]
        kwargs: dict[str, Any] = {
            "model": chosen_model,
            "messages": full_messages,
            "max_completion_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            if "parallel_tool_calls" not in self._unsupported_kwargs:
                kwargs["parallel_tool_calls"] = parallel_tool_calls
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        if self.reasoning_effort and "reasoning_effort" not in self._unsupported_kwargs:
            kwargs["reasoning_effort"] = self.reasoning_effort

        response = await self._call_with_fallback(kwargs)

        choice = response.choices[0]
        msg = choice.message
        text = msg.content or ""
        tool_calls: list[ToolCall] = []
        raw_tool_calls: list[dict[str, Any]] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = _parse_arguments(tc.function.arguments)
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
                raw_tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                )

        raw_assistant: dict[str, Any] = {"role": "assistant", "content": text or None}
        if raw_tool_calls:
            raw_assistant["tool_calls"] = raw_tool_calls

        usage_in = getattr(response.usage, "prompt_tokens", 0) or 0
        usage_out = getattr(response.usage, "completion_tokens", 0) or 0
        self.stats.calls += 1
        self.stats.input_tokens += usage_in
        self.stats.output_tokens += usage_out
        per_model = self.stats.per_model.setdefault(
            chosen_model, {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        )
        per_model["input_tokens"] += usage_in
        per_model["output_tokens"] += usage_out
        per_model["calls"] += 1

        return CallResult(
            text=text,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            raw_assistant_message=raw_assistant,
            input_tokens=usage_in,
            output_tokens=usage_out,
        )

    async def _call_with_fallback(self, kwargs: dict[str, Any]) -> Any:
        """Retry once after stripping any kwarg the API/model rejects."""
        try:
            return await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            message = str(exc)
            stripped: list[str] = []
            for key in ("reasoning_effort", "parallel_tool_calls"):
                if key in kwargs and key in message:
                    self._unsupported_kwargs.add(key)
                    kwargs.pop(key)
                    stripped.append(key)
            # max_completion_tokens too large for the chosen model → clamp.
            if "max_tokens is too large" in message or "max_completion_tokens" in message:
                kwargs["max_completion_tokens"] = min(
                    kwargs.get("max_completion_tokens", 4096), 4096
                )
                stripped.append("max_completion_tokens(clamped)")
            if not stripped:
                raise
            logger.warning("retrying after stripping/clamping kwargs %s: %s", stripped, exc)
            return await self._client.chat.completions.create(**kwargs)

    async def aclose(self) -> None:
        await self._client.close()

    def close(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
