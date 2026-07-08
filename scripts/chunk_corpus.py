"""Chunk the fetched NASA corpus (data/raw/) into data/processed/chunks.jsonl."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from app.ingestion.chunking import Chunk, chunk_document

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MANIFEST_PATH = RAW_DIR / "manifest.json"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"


def print_summary(all_chunks: list[Chunk], doc_count: int) -> None:
    token_counts = [c.token_count for c in all_chunks]

    print(f"Total docs: {doc_count}")
    print(f"Total chunks: {len(all_chunks)}")
    if not token_counts:
        return
    print(
        f"Token count distribution: min={min(token_counts)} "
        f"median={statistics.median(token_counts):.0f} max={max(token_counts)}"
    )

    largest = sorted(all_chunks, key=lambda c: c.token_count, reverse=True)[:5]
    smallest = sorted(all_chunks, key=lambda c: c.token_count)[:5]

    print("\n5 largest chunks:")
    for c in largest:
        print(f"  {c.token_count:>4} tok | {c.metadata.title[:60]} ({c.metadata.source_url})")

    print("\n5 smallest chunks:")
    for c in smallest:
        print(f"  {c.token_count:>4} tok | {c.metadata.title[:60]} ({c.metadata.source_url})")


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks: list[Chunk] = []
    for entry in manifest:
        raw_text = (RAW_DIR / entry["local_filename"]).read_text(encoding="utf-8")
        chunks = chunk_document(
            doc_id=entry["id"],
            raw_text=raw_text,
            title=entry["title"],
            source_url=entry["source_url"],
            source_family=entry["source_family"],
        )
        all_chunks.extend(chunks)

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk.model_dump_json())
            f.write("\n")

    print_summary(all_chunks, len(manifest))


if __name__ == "__main__":
    main()
