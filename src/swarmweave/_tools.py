"""Helpers that turn Python callables into OpenAI tool specs.

This module is intentionally small. We support plain Python callables that
have a docstring (used as the tool description) and typed parameters from a
short list of primitives. Anything more exotic should be expressed as a
hand-written tool dict and passed through.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

_PRIMITIVE_SCHEMA: dict[type, dict[str, str]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    list: {"type": "array"},
    dict: {"type": "object"},
}


def callable_to_tool_spec(func: Callable[..., Any]) -> dict[str, Any]:
    """Return an OpenAI-compatible tool spec for ``func``.

    The function's name becomes the tool name, the first paragraph of its
    docstring becomes the description, and its annotated parameters become
    the JSON schema. Parameters without defaults are marked required.
    """
    name = getattr(func, "__name__", "tool").replace(".", "_")
    doc = inspect.getdoc(func) or f"Call the {name} tool."
    description = doc.split("\n\n", 1)[0].strip()

    sig = inspect.signature(func)
    hints = get_type_hints(func)

    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(param_name, str)
        schema = _PRIMITIVE_SCHEMA.get(annotation, {"type": "string"})
        properties[param_name] = {**schema, "description": f"Argument '{param_name}'."}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def tool_spec_name(spec: dict[str, Any]) -> str:
    """Return the function name embedded in an OpenAI tool spec."""
    return str(spec.get("function", {}).get("name", ""))


def build_tool_registry(
    tools: list[Callable[..., Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Callable[..., Any]]]:
    """Return OpenAI tool specs and a name->callable lookup."""
    if not tools:
        return [], {}
    specs: list[dict[str, Any]] = []
    registry: dict[str, Callable[..., Any]] = {}
    for tool in tools:
        if isinstance(tool, str):
            raise NotImplementedError(
                "MCP tool URLs are a v0.2 surface. Pass Python callables for now; "
                "see docs/api_reference.md for the supported tool signature."
            )
        spec = callable_to_tool_spec(tool)
        specs.append(spec)
        registry[tool_spec_name(spec)] = tool
    return specs, registry
