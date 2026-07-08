"""Structure-aware chunking for NASA corpus documents.

Design (see CLAUDE.md "Tech decisions"): documents are first split on
structural boundaries (headings, paragraph breaks), then broken into
sentences so no chunk boundary ever falls mid-sentence. Sentences are packed
greedily, in document order, into chunks that target 300-500 tokens
(cl100k_base, via tiktoken), carrying ~50 tokens of trailing context forward
as overlap into the next chunk. This is a deliberate alternative to
embedding-based semantic chunking, which is out of scope until Phase 2.
"""

from __future__ import annotations

import hashlib
import re

import tiktoken
from pydantic import BaseModel

MIN_CHUNK_TOKENS = 300
MAX_CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50

_ENCODING = tiktoken.get_encoding("cl100k_base")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(“])")


class ChunkMetadata(BaseModel):
    title: str
    source_url: str
    source_family: str
    section_heading: str | None = None


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    token_count: int
    metadata: ChunkMetadata


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def split_into_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter: split after ./!/? before a new capitalized/quoted token."""
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_BOUNDARY_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def parse_segments(raw_text: str) -> list[tuple[str | None, str]]:
    """Split raw text into (current_heading, paragraph_text) pairs, in document order.

    Lines matching '#'*1-6 + heading text update the current heading and are not
    themselves emitted as content segments.
    """
    blocks = [b.strip() for b in raw_text.split("\n\n") if b.strip()]
    segments: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    for block in blocks:
        match = _HEADING_RE.match(block)
        if match:
            current_heading = match.group(2).strip()
            continue
        segments.append((current_heading, block))
    return segments


def _flatten_to_sentences(segments: list[tuple[str | None, str]]) -> list[tuple[str, str | None]]:
    flat: list[tuple[str, str | None]] = []
    for heading, paragraph in segments:
        for sentence in split_into_sentences(paragraph):
            flat.append((sentence, heading))
    return flat


def _make_chunk_id(doc_id: str, index: int) -> str:
    digest = hashlib.sha256(f"{doc_id}:{index}".encode("utf-8")).hexdigest()
    return digest[:16]


def chunk_document(
    doc_id: str,
    raw_text: str,
    title: str,
    source_url: str,
    source_family: str,
    min_tokens: int = MIN_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk a single document's raw text into structure-aware, token-budgeted chunks."""
    segments = parse_segments(raw_text)
    sentences = _flatten_to_sentences(segments)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[tuple[str, str | None]] = []
    current_tokens = 0

    def emit(pending: list[tuple[str, str | None]]) -> None:
        text = " ".join(sentence for sentence, _ in pending)
        heading = pending[0][1]
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(doc_id, len(chunks)),
                doc_id=doc_id,
                text=text,
                token_count=count_tokens(text),
                metadata=ChunkMetadata(
                    title=title,
                    source_url=source_url,
                    source_family=source_family,
                    section_heading=heading,
                ),
            )
        )

    for sentence, heading in sentences:
        sentence_tokens = count_tokens(sentence)

        if current and current_tokens >= min_tokens and current_tokens + sentence_tokens > max_tokens:
            emit(current)

            # Carry trailing sentences forward as overlap context for the next chunk.
            overlap: list[tuple[str, str | None]] = []
            overlap_token_total = 0
            for prev_sentence, prev_heading in reversed(current):
                prev_tokens = count_tokens(prev_sentence)
                if overlap_token_total + prev_tokens > overlap_tokens:
                    break
                overlap.insert(0, (prev_sentence, prev_heading))
                overlap_token_total += prev_tokens

            current = overlap
            current_tokens = overlap_token_total

        current.append((sentence, heading))
        current_tokens += sentence_tokens

    if current:
        emit(current)

    return chunks
