"""Integration tests for POST /chat: real OpenAI + real local DB.

Marked both `integration` (needs the local DB) and `llm` (makes real gpt-4o-mini
calls) - skipped if either is unavailable. The live-tool case mocks the NASA
HTTP call at the request_json boundary (we're testing the chat pipeline, not
NASA uptime) rather than via respx: respx's global transport patching turns
out to corrupt the OpenAI SDK's own (unrelated) HTTP responses when both are
active in the same process, even in pass-through mode with no matching route -
confirmed by reproducing it with zero registered routes. Patching
apod.request_json directly sidesteps that entirely. The retrieval case hits
the real DB and real embeddings/rerank calls.
"""

from __future__ import annotations

import json
import re

import httpx
import psycopg
import pytest

from app.config import settings
from app.main import app
from app.tools import apod as apod_module

pytestmark = [pytest.mark.integration, pytest.mark.llm]


@pytest.fixture(autouse=True)
def _stub_live_tools():
    """Override tests/agent/conftest.py's registry-level tool stub.

    That fixture fakes the whole tool (no HTTP at all), which is right for
    routing tests but wrong here: this file wants the real apod code path to
    run, with only the NASA HTTP call itself mocked via respx.
    """
    yield


def _db_reachable() -> bool:
    try:
        with psycopg.connect(settings.DATABASE_URL, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return True
    except psycopg.OperationalError:
        return False


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_name = None
        data_lines = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        events.append((event_name, json.loads("\n".join(data_lines))))
    return events


@pytest.mark.timeout(120)
async def test_chat_retrieval_query_streams_cited_answer():
    if not settings.OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set; skipping /chat integration test.")
    if not _db_reachable():
        pytest.skip("Postgres is not reachable; skipping /chat integration test.")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=60.0) as client:
        async with client.stream(
            "POST", "/chat", json={"query": "What did Webb discover about the Cigar Galaxy?"}
        ) as response:
            assert response.status_code == 200
            body = await response.aread()

    events = _parse_sse(body.decode())
    names = [name for name, _ in events]

    assert names[0] == "meta"
    assert names[1] == "sources"
    assert names[-1] == "done"
    assert all(name in ("meta", "sources", "delta", "done") for name in names)

    sources = next(data for name, data in events if name == "sources")
    assert len(sources) > 0

    full_text = "".join(data["text"] for name, data in events if name == "delta")
    assert re.search(r"\[\d+\]", full_text), f"expected an inline citation, got: {full_text!r}"

    done = next(data for name, data in events if name == "done")
    assert "token_usage" in done
    assert done["token_usage"]["total_tokens"] > 0


@pytest.mark.timeout(60)
async def test_chat_live_tool_query_routes_to_tools(monkeypatch):
    if not settings.OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set; skipping /chat integration test.")
    if not _db_reachable():
        pytest.skip("Postgres is not reachable; skipping /chat integration test.")

    async def fake_request_json(method, url, **kwargs):
        return {
            "title": "Test Nebula",
            "date": "2026-07-01",
            "explanation": "A lovely nebula.",
            "media_type": "image",
            "url": "https://apod.nasa.gov/apod/image/test.jpg",
        }

    monkeypatch.setattr(apod_module, "request_json", fake_request_json)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=60.0) as client:
        async with client.stream(
            "POST", "/chat", json={"query": "What is today's astronomy picture of the day?"}
        ) as response:
            assert response.status_code == 200
            body = await response.aread()

    events = _parse_sse(body.decode())
    meta = next(data for name, data in events if name == "meta")
    assert meta["route"] == "tools"
