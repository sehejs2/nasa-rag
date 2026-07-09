"""SSE event stream assembly for POST /chat.

Separated from app/main.py so the event sequence is directly unit-testable
(the wiring in main.py is a thin StreamingResponse wrapper around
chat_event_stream). Event protocol, in order: meta -> sources -> delta* -> done,
or error in place of/instead of any later event on failure.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Request

from app.agent.composer import compose_answer, validate_citations
from app.agent.loop import run_agent
from app.agent.sources import build_sources


def format_sse_event(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def chat_event_stream(query: str, request: Request | None = None) -> AsyncIterator[str]:
    """request is an optional FastAPI Request, used to poll for client disconnect
    mid-stream so we stop paying for tokens nobody is reading. None (e.g. in
    tests, or the manual CLI's in-process ASGI transport) just skips that check.
    """
    try:
        trace = await run_agent(query, mark_draft_superseded=True)
    except Exception as exc:  # noqa: BLE001 - surface any agent-loop failure as an error event
        yield format_sse_event("error", {"message": str(exc)})
        return

    yield format_sse_event(
        "meta",
        {
            "route": trace.route,
            "tools": sorted({tc.tool_name for tc in trace.tool_calls}),
            "iterations": trace.iterations_used,
        },
    )

    sources = build_sources(trace)
    yield format_sse_event("sources", [s.model_dump() for s in sources])

    full_text = ""
    composer_gen = compose_answer(query, trace, sources)
    try:
        async for delta in composer_gen:
            if request is not None and await request.is_disconnected():
                break
            full_text += delta
            yield format_sse_event("delta", {"text": delta})
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any mid-stream composition failure
        yield format_sse_event("error", {"message": str(exc)})
        return
    finally:
        await composer_gen.aclose()

    summary = validate_citations(full_text, sources)
    yield format_sse_event(
        "done",
        {
            "total_latency_ms": trace.total_latency_ms,
            "token_usage": trace.token_usage.model_dump(),
            "cited_sources": summary.cited_sources,
            "invalid_citations": summary.invalid_citations,
        },
    )
