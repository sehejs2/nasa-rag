.PHONY: dev test lint db-up db-down db-init corpus chunk ingest search tool ask chat eval frontend frontend-build docker-build docker-run

dev:
	uv run uvicorn app.main:app --reload

test:
	uv run pytest

lint:
	uv run ruff check .

db-up:
	docker compose up -d

db-down:
	docker compose down

db-init:
	uv run python -m app.ingestion.db

corpus:
	uv run python scripts/fetch_corpus.py

chunk:
	uv run python scripts/chunk_corpus.py

ingest:
	uv run python scripts/ingest.py

search:
	uv run python scripts/search_smoke.py "$(q)"

tool:
	uv run python scripts/run_tool.py "$(name)" '$(args)'

ask:
	uv run python scripts/ask.py "$(q)"

chat:
	uv run python scripts/chat_client.py "$(q)"

eval:
	uv run python scripts/run_eval.py $(args)

frontend:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

docker-build:
	docker build -t nasa-rag-backend .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env nasa-rag-backend
