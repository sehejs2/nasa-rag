from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.chat_stream import chat_event_stream
from app.agent.loop import run_agent
from app.retrieval.retriever import retrieve

TEXT_PREVIEW_CHARS = 300


class RetrieveRequest(BaseModel):
    query: str
    top_k: int | None = None


class AgentDebugRequest(BaseModel):
    query: str


class ChatRequest(BaseModel):
    query: str


def create_app() -> FastAPI:
    app = FastAPI(title="NASA RAG")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # Development/demo introspection endpoint: exposes raw retrieval internals
    # (per-stage timings, which reranker ran, similarity scores). No auth by
    # design for local debugging - must be disabled or protected before any
    # real deployment (Phase 9 concern).
    @app.post("/retrieve")
    async def retrieve_endpoint(body: RetrieveRequest):
        result = await retrieve(body.query, body.top_k)
        truncated_chunks = [
            chunk.model_copy(
                update={
                    "text": (
                        chunk.text[:TEXT_PREVIEW_CHARS] + "..."
                        if len(chunk.text) > TEXT_PREVIEW_CHARS
                        else chunk.text
                    )
                }
            )
            for chunk in result.chunks
        ]
        return result.model_copy(update={"chunks": truncated_chunks})

    # Development endpoint: exposes the full internal agent trace (every tool
    # call, arguments, token usage, the draft answer). No auth by design for
    # local debugging - must be disabled or protected before any real
    # deployment (Phase 9 concern), same as /retrieve.
    @app.post("/agent/debug")
    async def agent_debug_endpoint(body: AgentDebugRequest):
        return await run_agent(body.query)

    # Primary user-facing endpoint: runs the agent loop, then streams a cited
    # answer via SSE. See app/agent/chat_stream.py for the event protocol
    # (meta -> sources -> delta* -> done, or error) - documented in CLAUDE.md
    # as the contract the Phase 8 frontend builds against.
    @app.post("/chat")
    async def chat_endpoint(body: ChatRequest, request: Request):
        return StreamingResponse(
            chat_event_stream(body.query, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return app


app = create_app()
