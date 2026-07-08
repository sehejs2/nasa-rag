# NASA RAG System

Agentic RAG over NASA documents with live NASA API tools, built as a portfolio
project demonstrating LLM systems engineering (not framework wiring).

## Architecture

User query → Agent router (LLM decides: retrieve vs. live tool vs. both)
  → Path A: pgvector retrieval over embedded NASA mission reports/press releases, with reranking
  → Path B: Live NASA API tools (APOD, ISS Now, Mars Rover Photos, JWST)
  → LLM (GPT-4o-mini) composes a cited answer, streamed via SSE
  → Evaluation harness scores retrieval precision/recall and answer faithfulness

## Tech decisions (do not substitute without asking)

- Agent loop is HAND-ROLLED using the OpenAI SDK directly. No LangChain/LangGraph.
- Vector store: Postgres + pgvector (local via docker-compose, pgvector/pgvector:pg16).
- Embeddings: OpenAI text-embedding-3-small.
- Answer model: gpt-4o-mini.
- Chunking: structure-aware, not fixed-length or embedding-based semantic chunking.
  Documents are split on structural boundaries (headings, paragraph breaks) first,
  then packed sentence-by-sentence into chunks targeting 300-500 tokens (tiktoken,
  cl100k_base) with ~50 tokens of trailing overlap carried into the next chunk.
  Chunks never split mid-sentence. Chosen over embedding-based semantic chunking
  because embeddings are a Phase 2 concern — Phase 1 has no model calls, so
  structure is the only signal available for finding good split points.
- Streaming: Server-Sent Events from FastAPI.
- Package management: uv. Run commands via Makefile targets.

## Build phases

0. Scaffold (this file, FastAPI skeleton, docker-compose) — DONE
1. Corpus ingestion + structure-aware chunking (no embeddings yet) — DONE
2. Embedding pipeline + pgvector storage, idempotent ingestion CLI
3. Retrieval + reranking, /retrieve debug endpoint
4. NASA tool layer with function-calling schemas, mocked tests
5. Agent router loop + routing test set (~15 labeled queries)
6. Answer composition with inline citations + SSE streaming /chat
7. Eval harness: 30-50 labeled Qs, precision/recall + faithfulness, `make eval`
8. Next.js frontend: streamed chat, citations, retrieval-vs-tool badge
9. Deploy: backend + Postgres on Railway/Render, frontend on Vercel

## Conventions

- Python 3.11+, type hints everywhere, async where it matters (HTTP, DB, LLM calls).
- Tests in tests/, mirror the app/ package structure. Mock all external APIs in tests.
- Never commit .env or anything in data/. Secrets only via environment variables.
- Citations reference chunk IDs (retrieval) or tool-call IDs (live tools) so the
  frontend can render provenance.
- After each phase: run `make test` and `make lint`, then commit with a
  descriptive message. One phase = one or a few commits, no giant squashes.

## Commands

- make dev — run API with reload
- make test — pytest
- make lint — ruff
- make db-up / make db-down — local pgvector Postgres
- make corpus — fetch NASA documents into data/raw/ (see scripts/fetch_corpus.py)
- make chunk — chunk data/raw/ into data/processed/chunks.jsonl (see scripts/chunk_corpus.py)
