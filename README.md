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
9. Deploy: backend + Postgres on Railway, frontend on Vercel

## Deployment

**Architecture**: the FastAPI backend runs as a Docker container (see
`Dockerfile`) on [Railway](https://railway.app), backed by a Railway-managed
Postgres instance with the `pgvector` extension. The Next.js frontend deploys
separately to [Vercel](https://vercel.com). Railway builds the container
straight from the repo's `Dockerfile` — no `railway.json` or other
platform-specific config is needed; the Dockerfile is the contract.

- Backend: `https://<placeholder>.up.railway.app`
- Frontend: `https://<placeholder>.vercel.app`

### Environment variables

**Backend (Railway)**

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Embeddings, reranking, agent/composition, and eval-judge model calls. |
| `NASA_API_KEY` | Required by the `apod` and `mars_rover_photos` tools. |
| `DATABASE_URL` | Railway-provided Postgres connection string (`postgresql://...`; a legacy `postgres://` form is normalized automatically — see `app/config.py`). |
| `CORS_ORIGINS` | Comma-separated list of allowed frontend origins, e.g. the production Vercel URL. Defaults to `http://localhost:3000`. |

**Frontend (Vercel)**

| Variable | Purpose |
| --- | --- |
| `NEXT_PUBLIC_API_URL` | Base URL of the deployed backend, e.g. `https://<placeholder>.up.railway.app`. |

### One-time production DB bootstrap

Schema init and ingestion are plain CLI scripts that read `DATABASE_URL` from
the environment — run them from your laptop pointed at the production
database, no platform shell access needed:

```bash
DATABASE_URL="<railway-postgres-url>" make db-init   # creates the vector extension + schema
DATABASE_URL="<railway-postgres-url>" make ingest     # embeds and upserts the corpus
```

`make db-init` is idempotent (safe to re-run) and `make ingest` only calls the
embeddings API for chunks that are new or changed, so re-running either after
the corpus changes is cheap.
