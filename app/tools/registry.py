"""Registry of NASA tools for OpenAI function-calling and manual invocation."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.tools import apod, iss_now, jwst_images, mars_rover_photos, search_documents
from app.tools.base import ToolResult

_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[..., Awaitable[ToolResult]]


_TOOLS: dict[str, ToolSpec] = {
    module.NAME: ToolSpec(
        name=module.NAME,
        description=module.DESCRIPTION,
        parameters=module.PARAMETERS,
        run=module.run,
    )
    for module in (apod, iss_now, mars_rover_photos, jwst_images, search_documents)
}


def get_openai_tools() -> list[dict[str, Any]]:
    """Return tool schemas formatted for the OpenAI `tools=` chat completion param."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in _TOOLS.values()
    ]


def _matches_type(value: Any, expected_type: str) -> bool:
    python_type = _TYPE_MAP.get(expected_type)
    if python_type is None:
        return True
    if expected_type == "integer" and isinstance(value, bool):
        return False  # bool is an int subclass; don't let True/False pass as integer
    return isinstance(value, python_type)


def _validate_args(spec: ToolSpec, args: dict[str, Any]) -> str | None:
    """Return an error message if args don't satisfy the schema, else None."""
    schema = spec.parameters
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    missing = [name for name in required if name not in args]
    if missing:
        return f"Missing required argument(s): {', '.join(missing)}"

    if not schema.get("additionalProperties", True):
        unknown = [name for name in args if name not in properties]
        if unknown:
            return f"Unknown argument(s): {', '.join(unknown)}"

    for name, value in args.items():
        prop = properties.get(name)
        if prop is None:
            continue

        expected_type = prop.get("type")
        if expected_type and not _matches_type(value, expected_type):
            return f"Argument '{name}' must be of type {expected_type}, got {type(value).__name__}"

        enum = prop.get("enum")
        if enum is not None and value not in enum:
            return f"Argument '{name}' must be one of {enum}, got {value!r}"

        pattern = prop.get("pattern")
        if pattern and isinstance(value, str) and not re.fullmatch(pattern, value):
            return f"Argument '{name}' does not match required pattern {pattern}"

    return None


async def execute(name: str, args: dict[str, Any] | None = None) -> ToolResult:
    """Validate args against the tool's schema, run it, and always return a ToolResult."""
    args = args or {}
    spec = _TOOLS.get(name)
    if spec is None:
        return ToolResult(
            tool_name=name,
            ok=False,
            error=f"Unknown tool: {name!r}. Known tools: {sorted(_TOOLS)}",
            latency_ms=0.0,
            called_with=args,
        )

    error = _validate_args(spec, args)
    if error is not None:
        return ToolResult(tool_name=name, ok=False, error=error, latency_ms=0.0, called_with=args)

    return await spec.run(**args)
