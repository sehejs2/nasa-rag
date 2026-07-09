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
- Streaming: Server-Sent Events from FastAPI, hand-rolled (no sse-starlette) -
  event formatting is a single `format_sse_event()` string-join in
  app/agent/chat_stream.py, not worth a dependency for.
- Package management: uv. Run commands via Makefile targets.
- Idempotent ingestion: each chunk row stores a content_hash (sha256 of chunk
  text). A chunk is only sent to the embeddings API if it's new, its
  content_hash differs from what's stored (text changed upstream), or a prior
  run left its embedding NULL (crashed/failed before finishing). Unchanged
  chunks are skipped with zero API calls and zero DB writes. Chunk rows are
  upserted before embedding, with embedding forced to NULL whenever
  content_hash changes, so `make ingest` can be killed at any point and
  re-run safely — no duplicate rows, no silently-dropped failures.
- Retrieval is two-stage: stage 1 (recall) embeds the query and pulls the top 20
  candidates from pgvector by cosine similarity; stage 2 (precision) reranks
  those 20 down to top_k. Reranking sits behind a small Reranker interface
  (app/retrieval/reranker.py) so the strategy is swappable: LLMReranker (default)
  makes one batched gpt-4o-mini call scoring all candidates 0-10 as strict JSON,
  with one retry on malformed output before falling back to NoopReranker (vector
  order). A cross-encoder reranker could implement the same interface and be
  compared against LLMReranker once the Phase 7 eval harness exists to score
  retrieval quality objectively rather than by feel.

## Tools

Live NASA API tools (app/tools/), each with an OpenAI function-calling schema,
a typed async implementation, and a fresh-per-call httpx client (10s timeout,
2 retries with backoff on timeouts/5xx, no retry on 4xx):

- apod — NASA's Astronomy Picture of the Day for a given (or today's) date. Needs NASA_API_KEY.
- iss_now — current real-time ISS lat/lon position. No key needed.
- mars_rover_photos — Perseverance/Curiosity photos by Earth date or sol (or latest if neither given). Needs NASA_API_KEY.
- jwst_images — JWST-tagged media/caption search via the NASA Image and Video Library (no key needed); JWST *science* questions go to the RAG corpus instead, not this tool.
- search_documents — the Phase 3 retriever wrapped as a tool; searches the embedded corpus for missions, discoveries, science results, and history. NOT for real-time data (that's the four tools above).

Convention: tools never raise into the caller. Every tool catches its own
exceptions and returns `ToolResult(ok=False, error=...)`; the registry
(app/tools/registry.py) does the same for unknown tool names or invalid args.
The Phase 5 agent loop can treat every tool call as data, no try/except needed
around calls into the tool layer.

## Agent

The agent loop (app/agent/loop.py) routes via native OpenAI function calling
over the unified tool registry, not a hand-written classifier: retrieval is
just another entry in the same registry as the four live NASA tools
(search_documents alongside apod/iss_now/mars_rover_photos/jwst_images), so
gpt-4o-mini picks whichever tools (zero, one, or several, in parallel) a query
needs using the same tool-calling mechanism throughout. The loop is a plain
hand-rolled message loop over the OpenAI SDK (system + user messages, append
tool results as tool-role messages, repeat until the model returns
content with no tool calls, or a max-iteration guard forces one final
no-tools completion). Every tool result — including ok=False failures — is
serialized back to the model as data rather than raised, so a failed live API
call just becomes something the model reasons about (retry differently, use
another tool, or answer without it) instead of crashing the request. Route
("retrieval" | "tools" | "both" | "direct") is derived after the fact from
which tool names were actually called, not decided up front.

## Chat: composition + streaming (Phase 6)

POST /chat runs the agent loop (`run_agent(..., mark_draft_superseded=True)` -
the trace's own draft_answer is discarded; /agent/debug and `make ask` don't
pass this flag, so their output is unchanged), assembles a numbered source
list (app/agent/sources.py: chunks from search_documents dedup'd by chunk_id
first, then one source per successful live tool call, in call order - failed
tool calls never become sources), then streams a fresh gpt-4o-mini completion
(app/agent/composer.py) that's told to answer ONLY from those numbered sources
and cite claims inline as `[1]` / `[2][3]`. Citations are validated
(`validate_citations`) after the stream ends against the actual source
numbers that exist - invalid citation numbers are reported, never silently
rewritten into already-streamed text.

### SSE event protocol (app/agent/chat_stream.py)

Five event types, always in this order (or `error` in place of/instead of
anything after the point of failure):

1. `meta` — `{"route", "tools": [names], "iterations"}` - emitted as soon as the agent loop finishes, before composition starts.
2. `sources` — `[Source, ...]` (the full numbered list) - before any answer text, so the frontend can resolve citations live as they stream.
3. `delta` — `{"text": "..."}` - one per token/chunk of the composed answer.
4. `done` — `{"total_latency_ms", "token_usage", "cited_sources": [ints], "invalid_citations": [ints]}`.
5. `error` — `{"message": ...}` - agent-loop failures produce this before any `delta`; composition failures mid-stream produce it after whatever `delta`s already went out. The stream closes after `error` either way.

This is the exact contract the Phase 8 frontend builds against; scripts/chat_client.py (`make chat`) is the reference consumer.

## Eval harness (Phase 7)

`make eval` (scripts/run_eval.py) runs every case in `eval/cases.json` through
the real agent + composition pipeline and scores four things: routing
accuracy (trace.route vs expected_route, overall and per category), doc-level
retrieval metrics (precision@5/recall@5/MRR - chunks are deduplicated to
doc_ids since eval labels are doc-level, not chunk-level, so a human reviewer
can verify them by reading the doc), claim-level faithfulness (LLM-as-judge:
extract the answer's atomic claims, then verify each against the ACTUAL
sources the agent used that run; answers with zero claims, e.g. correct
declines, score `null` and are reported separately, not averaged in), and
key-facts answer completeness (does the answer semantically contain each
case's labeled key facts). `--rerank both` re-runs just the retrieval-scored
cases a second time with reranking off, for a side-by-side precision/recall/MRR
comparison. All four scorers are hand-rolled (no RAGAS or similar) - pure
functions for the metrics, gpt-4o-mini-as-judge (JUDGE_MODEL, swappable) for
faithfulness/key-facts, with one retry on malformed judge JSON before marking
a case `judge_error` rather than dropping it. Same-model-family judging
(gpt-4o-mini judging gpt-4o-mini's own answers) is known to run lenient - see
the caveat in app/evals/faithfulness.py. Every run writes a full-detail,
never-overwritten JSON artifact to `eval/results/<UTC timestamp>_results.json`
(git commit hash, run config, and per-case scores included) alongside the
console summary.

**`eval/cases.json` is a DRAFT** - 40 cases I generated by reading the actual
corpus, not yet reviewed by a human. See `eval/REVIEW.md` for the review
table. Do not treat reported numbers as validated until that review happens.

## Build phases

0. Scaffold (this file, FastAPI skeleton, docker-compose) — DONE
1. Corpus ingestion + structure-aware chunking (no embeddings yet) — DONE
2. Embedding pipeline + pgvector storage, idempotent ingestion CLI — DONE
3. Retrieval + reranking, /retrieve debug endpoint — DONE
4. NASA tool layer with function-calling schemas, mocked tests — DONE
5. Agent router loop + routing test set (~15 labeled queries) — DONE
6. Answer composition with inline citations + SSE streaming /chat — DONE
7. Eval harness: 30-50 labeled Qs, precision/recall + faithfulness, `make eval` — DONE (eval set UNVERIFIED, pending human review)
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
- make db-init — create/upgrade the pgvector schema (documents, chunks, HNSW index)
- make ingest — embed chunks.jsonl and upsert into Postgres, idempotently (see scripts/ingest.py)
- make search q="..." — debug tool: top-5 cosine-similarity matches for a query (see scripts/search_smoke.py)
- make tool name=<tool> args='{"key": "value"}' — manually invoke a NASA tool (see scripts/run_tool.py)
- make ask q="..." — run the agent loop end-to-end and print the trace + draft answer (see scripts/ask.py)
- make chat q="..." — run POST /chat end-to-end and stream the cited answer + sources (see scripts/chat_client.py)
- make eval args="..." — run the eval harness (see scripts/run_eval.py; flags: --cases, --rerank on|off|both, --limit N, --concurrency)
