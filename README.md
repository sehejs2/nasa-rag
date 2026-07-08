# NASA RAG System

Agentic RAG over NASA mission reports and press releases, combined with live
NASA API tools (APOD, ISS Now, Mars Rover Photos, JWST). A hand-rolled agent
loop routes each query to pgvector retrieval, a live tool, or both, then
composes a cited, streamed answer with GPT-4o-mini. Scored by a custom
evaluation harness. Built as a portfolio project in LLM systems engineering.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Docker (for local Postgres + pgvector)

## Setup

```bash
cp .env.example .env   # fill in OPENAI_API_KEY and NASA_API_KEY
make db-up              # start pgvector Postgres via docker-compose
uv sync                 # install dependencies
make dev                # run the API with reload
```

Visit `http://localhost:8000/health` to confirm it's running.

## Roadmap

0. Scaffold: FastAPI skeleton, docker-compose, CLAUDE.md — done
1. Corpus ingestion + semantic chunking (no embeddings yet)
2. Embedding pipeline + pgvector storage, idempotent ingestion CLI
3. Retrieval + reranking, `/retrieve` debug endpoint
4. NASA tool layer with function-calling schemas, mocked tests
5. Agent router loop + routing test set (~15 labeled queries)
6. Answer composition with inline citations + SSE streaming `/chat`
7. Eval harness: 30-50 labeled questions, precision/recall + faithfulness
8. Next.js frontend: streamed chat, citations, retrieval-vs-tool badge
9. Deploy: backend + Postgres on Railway/Render, frontend on Vercel
