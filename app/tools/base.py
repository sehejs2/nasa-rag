"""Shared infrastructure for NASA API tools: result contract, retrying HTTP client.

Tools never raise into the caller - every tool catches its own exceptions and
returns ToolResult(ok=False, error=...), so the Phase 5 agent loop can treat a
failed tool call as data rather than needing its own try/except around every call.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings

REQUEST_TIMEOUT_SECONDS = 10.0
MAX_RETRIES = 2  # additional attempts beyond the first, on timeouts/5xx only


class ToolResult(BaseModel):
    tool_name: str
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: float
    called_with: dict[str, Any]


def require_nasa_api_key() -> str:
    """Return the configured NASA_API_KEY, or raise if it's unset.

    Tools that need it call this first; the exception is caught by run_tool()
    and turned into a clean ToolResult(ok=False) rather than a raw crash.
    """
    if not settings.NASA_API_KEY:
        raise RuntimeError(
            "NASA_API_KEY is not set. Get a free key at https://api.nasa.gov and add it to .env."
        )
    return settings.NASA_API_KEY


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(1 + MAX_RETRIES),
    reraise=True,
)
async def _request(client: httpx.AsyncClient, method: str, url: str, **kwargs: Any) -> httpx.Response:
    response = await client.request(method, url, **kwargs)
    response.raise_for_status()
    return response


async def request_json(method: str, url: str, **kwargs: Any) -> Any:
    """GET/POST returning parsed JSON.

    Retries timeouts and 5xx responses up to MAX_RETRIES times with exponential
    backoff; 4xx responses raise immediately with no retry. A fresh client is
    opened and closed per call rather than shared/cached, so nothing can be
    constructed in one event loop and reused (and deadlock) in another's.
    """
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = await _request(client, method, url, **kwargs)
        return response.json()


async def run_tool(
    tool_name: str,
    called_with: dict[str, Any],
    fetch: Callable[[], Awaitable[dict[str, Any]]],
) -> ToolResult:
    """Time a tool's fetch and convert any exception into ok=False - tools never raise."""
    start = time.perf_counter()
    try:
        data = await fetch()
    except Exception as exc:  # noqa: BLE001 - deliberately broad: tools must never raise
        return ToolResult(
            tool_name=tool_name,
            ok=False,
            error=str(exc) or type(exc).__name__,
            latency_ms=(time.perf_counter() - start) * 1000,
            called_with=called_with,
        )
    return ToolResult(
        tool_name=tool_name,
        ok=True,
        data=data,
        latency_ms=(time.perf_counter() - start) * 1000,
        called_with=called_with,
    )
