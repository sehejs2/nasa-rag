from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.retrieval.retriever import retrieve

TEXT_PREVIEW_CHARS = 300


class RetrieveRequest(BaseModel):
    query: str
    top_k: int | None = None


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

    return app


app = create_app()
