"""Document retrieval (Phase 3 two-stage retriever), wrapped as an agent tool.

ToolResult.data carries every field RetrievedChunk needs (chunk_id, doc_id,
title, source_url, source_family, section, similarity) so the agent loop can
reconstruct full RetrievedChunk objects for Phase 6 citations directly from it
-- only `text` is truncated, specifically to keep the message payload sent
back to the router model small. Citations key off chunk_id, not raw text, so
truncating text here costs nothing citation-relevant.
"""

from __future__ import annotations

from app.retrieval.retriever import retrieve
from app.tools.base import ToolResult, run_tool

NAME = "search_documents"
DESCRIPTION = (
    "Search the embedded corpus of NASA mission reports, JWST press releases, and "
    "Mars mission summaries. Use for questions about missions, discoveries, science "
    "results, and history. NOT for real-time data like current positions or today's "
    "picture - use the matching live tool (iss_now, apod, mars_rover_photos, "
    "jwst_images) for those instead."
)
PARAMETERS: dict = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "A natural-language search query describing the information needed.",
        }
    },
    "required": ["query"],
    "additionalProperties": False,
}

MESSAGE_TEXT_PREVIEW_CHARS = 500


def _truncate(text: str, limit: int = MESSAGE_TEXT_PREVIEW_CHARS) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


async def _fetch(query: str) -> dict:
    result = await retrieve(query)
    chunks = [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "source_url": chunk.source_url,
            "source_family": chunk.source_family,
            "section": chunk.section,
            "text": _truncate(chunk.text),
            "similarity": chunk.rerank_score,
        }
        for chunk in result.chunks
    ]
    return {"chunks": chunks}


async def run(query: str) -> ToolResult:
    return await run_tool(NAME, {"query": query}, lambda: _fetch(query))
