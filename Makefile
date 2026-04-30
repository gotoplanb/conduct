.PHONY: help install run worker migrate revision seed test lint format up down logs psql redis-cli tunnel

help:
	@echo "Common targets:"
	@echo "  make install   — sync deps via uv"
	@echo "  make up        — start postgres + redis containers"
	@echo "  make down      — stop containers"
	@echo "  make migrate   — run alembic upgrade head"
	@echo "  make revision m=\"message\" — create alembic autogenerate revision"
	@echo "  make run       — run the FastAPI app on :8000 with reload"
	@echo "  make worker    — run the RQ worker"
	@echo "  make seed      — bootstrap clients + routing rules (idempotent)"
	@echo "  make test      — run pytest"
	@echo "  make lint      — ruff check"
	@echo "  make format    — ruff format"
	@echo "  make psql      — open psql against the dev database"
	@echo "  make redis-cli — open redis-cli against the dev redis"
	@echo "  make tunnel    — start the ngrok HTTPS tunnel"

install:
	uv sync

up:
	docker compose up -d
	@echo "Waiting for postgres..."
	@until docker compose exec -T postgres pg_isready -U conduct >/dev/null 2>&1; do sleep 1; done
	@echo "Postgres ready."

down:
	docker compose down

migrate:
	uv run alembic upgrade head

revision:
	@if [ -z "$(m)" ]; then echo "Usage: make revision m=\"message\""; exit 1; fi
	uv run alembic revision --autogenerate -m "$(m)"

run:
	uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

worker:
	uv run python -m worker.queue

seed:
	uv run python -m scripts.seed

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

psql:
	docker compose exec postgres psql -U conduct -d conduct

redis-cli:
	docker compose exec redis redis-cli

tunnel:
	ngrok http 8000
