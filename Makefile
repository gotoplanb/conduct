.PHONY: help install run worker migrate revision seed test lint format up up-infra down build logs psql redis-cli tunnel

help:
	@echo "Common targets:"
	@echo "  make install   — sync deps via uv"
	@echo "  make up        — start the full stack (postgres, redis, api, worker)"
	@echo "  make up-infra  — start only postgres + redis (use with host-side make run / make worker)"
	@echo "  make down      — stop all containers"
	@echo "  make build     — rebuild the conduct image"
	@echo "  make migrate   — run alembic upgrade head (host-side, against the compose postgres)"
	@echo "  make revision m=\"message\" — create alembic autogenerate revision"
	@echo "  make run       — run the FastAPI app on :8000 with reload (host-side dev)"
	@echo "  make worker    — run the RQ worker (host-side dev)"
	@echo "  make seed      — bootstrap clients + routing rules (idempotent)"
	@echo "  make test      — run pytest"
	@echo "  make lint      — ruff check"
	@echo "  make format    — ruff format"
	@echo "  make psql      — open psql against the dev database"
	@echo "  make redis-cli — open redis-cli against the dev redis"
	@echo "  make tunnel    — start the ngrok HTTPS tunnel"
	@echo "  make download-voice [v=<voice-name>] — download a Piper TTS voice (default en_US-amy-medium)"
	@echo "  make sonar-scan — run SonarQube static analysis (results at http://localhost:9000/dashboard?id=conduct)"

install:
	uv sync

up:
	docker compose up -d
	@echo "Waiting for postgres..."
	@until docker compose exec -T postgres pg_isready -U conduct >/dev/null 2>&1; do sleep 1; done
	@echo "Postgres ready."

up-infra:
	docker compose up -d postgres redis
	@echo "Waiting for postgres..."
	@until docker compose exec -T postgres pg_isready -U conduct >/dev/null 2>&1; do sleep 1; done
	@echo "Postgres ready."

down:
	docker compose down

build:
	GIT_SHA=$$(git rev-parse HEAD 2>/dev/null || echo "") docker compose build api worker

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

# Run SonarQube static analysis against the local watchtower instance.
# Reads SONAR_TOKEN from .env. Results appear at http://localhost:9000/dashboard?id=conduct
sonar-scan:
	@if [ -z "$$SONAR_TOKEN" ] && ! grep -q '^SONAR_TOKEN=' .env 2>/dev/null; then \
		echo "Set SONAR_TOKEN in .env (generate at http://localhost:9000)"; exit 1; \
	fi
	@set -a; . ./.env; set +a; \
	docker run --rm \
		-e SONAR_HOST_URL=http://host.docker.internal:9000 \
		-e SONAR_TOKEN=$$SONAR_TOKEN \
		-v "$$(pwd):/usr/src" \
		sonarsource/sonar-scanner-cli:latest

# Download a Piper voice into ./voices. Override v=<voice-name> to grab a
# different one. Browse https://huggingface.co/rhasspy/piper-voices for the
# full catalog. The default en_US-amy-medium is ~25MB.
v ?= en_US-amy-medium
download-voice:
	@mkdir -p voices
	@LANG=$$(echo "$(v)" | cut -d_ -f1); \
	REGION=$$(echo "$(v)" | cut -d_ -f2 | cut -d- -f1); \
	NAME=$$(echo "$(v)" | cut -d- -f2); \
	QUALITY=$$(echo "$(v)" | cut -d- -f3); \
	BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/$$LANG/$${LANG}_$${REGION}/$$NAME/$${QUALITY}/$(v)"; \
	echo "Downloading $(v) from $$BASE..."; \
	curl -sSL "$$BASE.onnx" -o "voices/$(v).onnx" && \
	curl -sSL "$$BASE.onnx.json" -o "voices/$(v).onnx.json" && \
	echo "Done. Voice ready at voices/$(v).onnx"
