.PHONY: dev test lint db-up db-down

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
