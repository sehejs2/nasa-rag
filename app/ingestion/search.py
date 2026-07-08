"""Cosine-similarity search against the pgvector store.

This is the query used by the scripts/search_smoke.py debugging tool, not the
real retrieval module (that's Phase 3). Split out into its own function so it
can be exercised directly in integration tests.
"""

from __future__ import annotations

import psycopg

TOP_K = 5

# The query embedding arrives as a plain Python list[float] (that's what
# Embedder.embed_all returns), so it must be cast explicitly to `vector` -
# psycopg's default list dumper sends it as a `double precision[]` array, and
# there is no `vector <=> double precision[]` operator.
SEARCH_SQL = """
SELECT c.text, c.section, d.title, 1 - (c.embedding <=> %(query_embedding)s::vector) AS similarity
FROM chunks c
JOIN documents d ON d.doc_id = c.doc_id
WHERE c.embedding IS NOT NULL
ORDER BY c.embedding <=> %(query_embedding)s::vector
LIMIT %(limit)s
"""


def similarity_search(
    conn: psycopg.Connection, query_embedding: list[float], top_k: int = TOP_K
) -> list[tuple[str, str | None, str, float]]:
    """Return up to top_k (text, section, title, similarity) rows, most similar first."""
    return conn.execute(SEARCH_SQL, {"query_embedding": query_embedding, "limit": top_k}).fetchall()
