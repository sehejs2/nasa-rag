"""Plain-SQL schema and connection helpers for the pgvector store.

No ORM, no migration framework (Alembic etc.) by design — the schema is small
and stable enough that raw, inspectable SQL is easier to reason about than a
migration chain. `init_schema` is idempotent and safe to run on every startup.
"""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from app.config import settings

EMBEDDING_DIMENSIONS = 1536

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_family TEXT NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    section TEXT,
    content_hash TEXT NOT NULL,
    embedding VECTOR({EMBEDDING_DIMENSIONS}),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Approximate nearest-neighbor index for cosine similarity search. At this
-- corpus size (hundreds of chunks) an exact sequential scan would be plenty
-- fast; this index exists for scale-readiness as the corpus grows, not
-- because it's needed today.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def get_connection(database_url: str | None = None) -> psycopg.Connection:
    return psycopg.connect(database_url or settings.DATABASE_URL, autocommit=True)


def init_schema(conn: psycopg.Connection) -> None:
    """Create the schema if needed and register the vector type adapter.

    Idempotent: safe to call at the start of every script, not just once via
    `make db-init`. The vector type can only be registered with psycopg after
    the `vector` extension exists, so schema creation and type registration
    are bundled together here.
    """
    conn.execute(SCHEMA_SQL)
    register_vector(conn)


def main() -> None:
    conn = get_connection()
    try:
        init_schema(conn)
        print("Schema initialized (documents, chunks, HNSW index).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
