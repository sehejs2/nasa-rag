"""Cited answer composition: stream gpt-4o-mini tokens grounded ONLY in numbered sources.

compose_answer yields text deltas as they arrive; it does not itself validate
citations or rewrite streamed text (you can't un-send bytes already streamed
to a client). validate_citations is a separate, pure function the caller runs
once the full text is assembled, used to build the `done` SSE event.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.agent.models import AgentTrace
from app.agent.sources import Source
from app.config import settings

MODEL = "gpt-4o-mini"
# See app/tools/base.py / app/retrieval/reranker.py / app/agent/loop.py for why
# this is explicit rather than the SDK's 600s default.
OPENAI_REQUEST_TIMEOUT_SECONDS = 30.0

SOURCE_TEXT_CHARS = 800

CITATION_RE = re.compile(r"\[(\d+)\]")

COMPOSITION_INSTRUCTIONS = (
    "Answer the question using ONLY the numbered sources above - do not use outside "
    "knowledge. Cite claims inline with bracketed numbers like [1] or [2][3] "
    "immediately after the claim they support. If the sources don't contain the "
    "answer, say so plainly instead of inventing one. Keep the answer concise."
)


class CitationSummary(BaseModel):
    cited_sources: list[int]
    invalid_citations: list[int]


def _body_for_source(source: Source, trace: AgentTrace) -> str:
    """The actual payload text for a source, looked up from the trace by ref_id."""
    if source.kind == "chunk":
        chunk = next((c for c in trace.retrieved_chunks if c.chunk_id == source.ref_id), None)
        text = chunk.text if chunk else ""
        return text if len(text) <= SOURCE_TEXT_CHARS else text[:SOURCE_TEXT_CHARS].rstrip() + "..."

    record = next((tc for tc in trace.tool_calls if tc.call_id == source.ref_id), None)
    if record is None or not record.result_data:
        return ""
    return json.dumps(record.result_data)


def render_sources_block(sources: list[Source], trace: AgentTrace) -> str:
    """The numbered '[n] title — body' text block shared by composition and
    (Phase 7) faithfulness judging, so both see the exact same source content.
    """
    if not sources:
        return "(no sources were found for this question)"
    blocks = [f"[{s.number}] {s.title} — {_body_for_source(s, trace)}" for s in sources]
    return "\n\n".join(blocks)


def build_composition_messages(query: str, trace: AgentTrace, sources: list[Source]) -> list[dict]:
    sources_text = render_sources_block(sources, trace)
    user_content = f"Question: {query}\n\nSources:\n{sources_text}\n\n{COMPOSITION_INSTRUCTIONS}"
    return [{"role": "user", "content": user_content}]


async def compose_answer(
    query: str,
    trace: AgentTrace,
    sources: list[Source],
    *,
    client: AsyncOpenAI | None = None,
) -> AsyncIterator[str]:
    client = client or AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        timeout=OPENAI_REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )
    messages = build_composition_messages(query, trace, sources)

    stream = await client.chat.completions.create(model=MODEL, messages=messages, stream=True)
    try:
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            await close()


def validate_citations(text: str, sources: list[Source]) -> CitationSummary:
    valid_numbers = {s.number for s in sources}
    cited = sorted({int(n) for n in CITATION_RE.findall(text)})
    return CitationSummary(
        cited_sources=[n for n in cited if n in valid_numbers],
        invalid_citations=[n for n in cited if n not in valid_numbers],
    )
