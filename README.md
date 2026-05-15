<img src="hermit-conduct.png" alt="Conduct mascot — a hermit crab in a tuxedo holding a conductor's baton" width="200">

# Conduct

Multi-tenant LLM dispatch service. Routes AI workloads to local (Ollama) or cloud (Anthropic) models based on `task_type`, sensitivity, and availability. FastAPI + Postgres + Redis + RQ. See [SPEC.md](SPEC.md) for the design.

## Quickstart

```bash
make install                                                 # uv sync
cp .env.example .env                                         # fill in CONDUCT_ADMIN_KEY, ANTHROPIC_API_KEY
cp config/seed.clients.example.yaml config/seed.clients.yaml # then edit to add your clients
make up                                                      # full stack: postgres, redis, api, worker
make migrate                                                 # alembic upgrade head
make seed                                                    # creates clients + 8 routing rules. Prints raw API keys ONCE.
```

After `make seed`, save the printed client keys — they're hashed in the DB and unrecoverable.

For fast-iteration dev (uvicorn `--reload`), use `make up-infra` (postgres + redis only) followed by `make run` and `make worker` in separate terminals. See [docs/deployment.md](docs/deployment.md) for the trade-offs.

## What runs where

```
┌──────────────┐     ┌──────────────┐     ┌────────────────┐
│   API :8000  │────▶│  Postgres    │◀────│  Worker (RQ)   │
│              │     │  :5432       │     │                │
│ /metrics     │     └──────────────┘     │  metrics :8001 │
│ /prometheus  │     ┌──────────────┐     │                │
│              │────▶│   Redis      │◀────│                │
└──────┬───────┘     │   :6379      │     └────────┬───────┘
       │             └──────────────┘              │
       │                                           │
       │  OTLP gRPC                    OTLP gRPC   │
       └────────────▶  :4317  ◀──────────────────┘
                      (Watchtower's Alloy)
```

All four services run as containers via `docker-compose.yml`. Ollama stays on the host (Docker Desktop on macOS can't pass through Metal GPU access, which 70b-class models need). Watchtower's LGTM stack (Tempo/Loki/Mimir/Grafana) lives in `~/watchtower` as a separate Compose project.

## API surface

| Method  | Path                          | Auth   | Notes |
|---------|-------------------------------|--------|-------|
| POST    | `/jobs`                       | client | sync (cloud) or 202+enqueue (local or `"async":true`). Per-client rate limit |
| GET     | `/jobs/{id}`                  | client | owner-only |
| DELETE  | `/jobs/{id}`                  | client | cancel pending; 409 if running |
| POST    | `/clients`                    | admin  | returns raw API key once |
| GET     | `/clients`                    | admin  | |
| PATCH   | `/clients/{id}`               | admin  | |
| GET     | `/clients/{id}/usage?days=N`  | admin  | daily aggregates |
| GET     | `/models`                     | admin  | local from Ollama × cloud from `pricing.yaml` |
| POST    | `/models/{name}/load`         | admin  | |
| POST    | `/models/{name}/unload`       | admin  | |
| GET     | `/routing`                    | admin  | |
| PUT     | `/routing/{task_type}`        | admin  | hot-reload — no restart |
| GET     | `/metrics`                    | admin  | JSON aggregator with filters |
| GET     | `/metrics/prometheus`         | open   | scrape target for Alloy |
| GET     | `/eval/compare?task_type=X`   | admin  | per-model side-by-side |
| POST    | `/eval/jobs/{id}/score`       | admin  | manual quality rating 1–5 (job or shadow) |
| GET     | `/eval/review`                | admin  | unscored shadows for human rating |
| POST    | `/tts`                        | client | enqueue text→MP3 job; returns poll URL |
| GET     | `/output/{file}.mp3`          | admin  | serve generated audiobook chunks |
| GET     | `/health`                     | open   | DB ping |

## Read more

- **[docs/architecture.md](docs/architecture.md)** — sensitivity tiers, sync vs. async decision, routing engine, failure handling
- **[docs/deployment.md](docs/deployment.md)** — container build, host-side vs. containerized dev, git SHA provenance, private overlays, ECS / Cloud Run targets, ngrok
- **[docs/operations.md](docs/operations.md)** — live config, observability, common queries, tests, DoD
- **[docs/tts.md](docs/tts.md)** — text-to-speech (Piper) for audiobook-style workloads
- **[SPEC.md](SPEC.md)** — original design doc

## Project layout

```
main.py                    FastAPI entrypoint + router registration
lifespan.py                tracing setup, providers, SIGUSR1 pricing reload
auth.py                    Bearer auth (client + admin)
deps.py                    shared deps (provider registry from app.state)
rate_limit.py              per-client Redis tumbling-window limiter
prompt_loader.py           DB-backed prompt resolver (per-client override → shared)

config/                    settings + pricing
db/                        SQLAlchemy 2.0 async session + declarative base
models/                    ORM models + Sensitivity / JobStatus enums
providers/                 BaseProvider, Ollama, Anthropic, registry
routing/                   pure decide() with sensitivity floor
worker/                    queue, runner (RQ entry), executor (sync+async share this)
retry/                     FailureHandler interface (static v1, triage v2 stub)
observability/             OTel tracer + Prometheus metric helpers
routes/                    route modules; one per concern
prompts/                   seed material — shared/ + clients/{name}/ .md files
scripts/seed.py            idempotent bootstrap (clients, routing rules, prompts)
scripts/cli.py             `conduct` admin CLI (prompts list/get/edit/history)
tests/                     unit tests (pytest)
alembic/                   migrations
docs/                      verbose docs (architecture, deployment, operations)
Dockerfile                 multi-stage uv build; one image for api + worker
docker-compose.yml         postgres, redis, api, worker
```

---

© 2026 Zero Mission LLC
