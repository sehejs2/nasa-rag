"""Numbered source list assembly for cited answer composition (Phase 6).

Deterministic ordering: chunks first (by retrieval rank), then tools (by call
order). Failed tool calls never become sources - there's nothing to cite.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.agent.models import AgentTrace, ToolCallRecord

SourceKind = Literal["chunk", "tool"]

SEARCH_DOCUMENTS_TOOL = "search_documents"


class Source(BaseModel):
    number: int
    kind: SourceKind
    title: str
    url: str | None
    detail: str
    ref_id: str


def _tool_source_url(tool_name: str, result_data: dict | None) -> str | None:
    """Best-known URL for a live tool's result, else None. Tool-specific by design."""
    if not result_data:
        return None
    if tool_name == "apod":
        return result_data.get("url")
    if tool_name == "mars_rover_photos":
        photos = result_data.get("photos") or []
        return photos[0]["img_src"] if photos else None
    if tool_name == "jwst_images":
        items = result_data.get("items") or []
        return items[0]["image_url"] if items else None
    return None  # e.g. iss_now has no natural URL


def _tool_detail(record: ToolCallRecord) -> str:
    args_summary = ", ".join(f"{k}={v}" for k, v in record.arguments.items() if v is not None)
    return f"{record.tool_name}({args_summary})" if args_summary else f"{record.tool_name}()"


def build_sources(trace: AgentTrace) -> list[Source]:
    sources: list[Source] = []
    number = 1

    seen_chunk_ids: set[str] = set()
    for chunk in sorted(trace.retrieved_chunks, key=lambda c: c.rank):
        if chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.chunk_id)
        sources.append(
            Source(
                number=number,
                kind="chunk",
                title=chunk.title,
                url=chunk.source_url,
                detail=chunk.section or chunk.source_family,
                ref_id=chunk.chunk_id,
            )
        )
        number += 1

    for record in trace.tool_calls:
        if not record.result_ok or record.tool_name == SEARCH_DOCUMENTS_TOOL:
            continue
        sources.append(
            Source(
                number=number,
                kind="tool",
                title=record.tool_name,
                url=_tool_source_url(record.tool_name, record.result_data),
                detail=_tool_detail(record),
                ref_id=record.call_id,
            )
        )
        number += 1

    return sources
