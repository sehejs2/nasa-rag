"""Debug tool: embed a query, print the top-5 most similar chunks by cosine similarity.

Not the retrieval module (that's Phase 3) - just a way to eyeball whether the
embeddings and pgvector index are behaving sanely.
"""

from __future__ import annotations

import asyncio
import sys

from app.ingestion.db import get_connection, init_schema
from app.ingestion.embedder import Embedder

TOP_K = 5

SEARCH_SQL = """
SELECT c.text, c.section, d.title, 1 - (c.embedding <=> %(query_embedding)s) AS similarity
FROM chunks c
JOIN documents d ON d.doc_id = c.doc_id
WHERE c.embedding IS NOT NULL
ORDER BY c.embedding <=> %(query_embedding)s
LIMIT %(limit)s
"""


async def embed_query(query: str) -> list[float]:
    embeddings, failed = await Embedder().embed_all([("query", query)])
    if failed:
        raise RuntimeError("Failed to embed the query after retries.")
    return embeddings["query"]


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print('Usage: make search q="your question here"', file=sys.stderr)
        sys.exit(1)
    query = sys.argv[1]

    query_embedding = asyncio.run(embed_query(query))

    conn = get_connection()
    init_schema(conn)
    rows = conn.execute(SEARCH_SQL, {"query_embedding": query_embedding, "limit": TOP_K}).fetchall()
    conn.close()

    if not rows:
        print("No embedded chunks found. Run `make ingest` first.")
        return

    print(f'Top {len(rows)} matches for: "{query}"\n')
    for i, (text, section, title, similarity) in enumerate(rows, start=1):
        preview = text[:200].replace("\n", " ")
        print(f"{i}. similarity={similarity:.4f} | {title} | section={section}")
        print(f"   {preview}...\n")


if __name__ == "__main__":
    main()
