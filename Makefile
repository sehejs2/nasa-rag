.PHONY: dev test lint db-up db-down db-init corpus chunk ingest search

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
