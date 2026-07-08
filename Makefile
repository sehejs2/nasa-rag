.PHONY: dev test lint db-up db-down corpus chunk

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

corpus:
	uv run python scripts/fetch_corpus.py

chunk:
	uv run python scripts/chunk_corpus.py
