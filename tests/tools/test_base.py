"""Tests for shared tool infrastructure: retry/backoff and the ToolResult contract.

Covers the generic HTTP retry semantics once here; individual tool test files
focus on their own URL/param building and response parsing.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.tools.base import request_json, run_tool

TEST_URL = "https://example.com/api"


@respx.mock
async def test_request_json_happy_path():
    respx.get(TEST_URL).mock(return_value=httpx.Response(200, json={"ok": True}))

    data = await request_json("GET", TEST_URL)

    assert data == {"ok": True}


@respx.mock
async def test_timeout_retries_then_fails():
    route = respx.get(TEST_URL).mock(side_effect=httpx.TimeoutException("boom"))

    with pytest.raises(httpx.TimeoutException):
        await request_json("GET", TEST_URL)

    assert route.call_count == 3  # 1 initial + 2 retries


@respx.mock
async def test_timeout_then_succeeds_on_retry():
    route = respx.get(TEST_URL).mock(
        side_effect=[httpx.TimeoutException("boom"), httpx.Response(200, json={"ok": True})]
    )

    data = await request_json("GET", TEST_URL)

    assert data == {"ok": True}
    assert route.call_count == 2


@respx.mock
async def test_5xx_retries_then_fails():
    route = respx.get(TEST_URL).mock(return_value=httpx.Response(503))

    with pytest.raises(httpx.HTTPStatusError):
        await request_json("GET", TEST_URL)

    assert route.call_count == 3


@respx.mock
async def test_4xx_does_not_retry():
    route = respx.get(TEST_URL).mock(return_value=httpx.Response(404))

    with pytest.raises(httpx.HTTPStatusError):
        await request_json("GET", TEST_URL)

    assert route.call_count == 1


@respx.mock
async def test_malformed_json_body_raises():
    respx.get(TEST_URL).mock(return_value=httpx.Response(200, content=b"not json"))

    with pytest.raises(Exception):  # noqa: B017 - httpx/json raise different types across versions
        await request_json("GET", TEST_URL)


async def test_run_tool_wraps_success():
    async def fetch():
        return {"foo": "bar"}

    result = await run_tool("t", {"a": 1}, fetch)

    assert result.ok is True
    assert result.data == {"foo": "bar"}
    assert result.tool_name == "t"
    assert result.called_with == {"a": 1}
    assert result.latency_ms >= 0
    assert result.error is None


async def test_run_tool_wraps_exception_as_ok_false():
    async def fetch():
        raise ValueError("kaboom")

    result = await run_tool("t", {}, fetch)

    assert result.ok is False
    assert "kaboom" in result.error
    assert result.data is None
