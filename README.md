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

**Want to drive Conduct from your phone?** [docs/quickstart-ios-eval.md](docs/quickstart-ios-eval.md) walks through firing a dad joke from the Claude iOS app, pulling every model's attempt side-by-side, and scoring the best one — without leaving the chat.

**Wiring up cloud providers?** [docs/mcp-connector.md](docs/mcp-connector.md) covers the OAuth + tool surface. Per-client Anthropic API keys are described in the iOS quickstart above; AWS Bedrock setup (IAM, model IDs, pricing) lives in [docs/bedrock.md](docs/bedrock.md).

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
| GET     | `/jobs`                       | admin  | list across clients; filters: `task_type`, `status`, `q`, `limit` |
| GET     | `/jobs/{id}`                  | client/admin | owner sees own; admin sees any |
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
| GET     | `/.well-known/oauth-*`        | open   | OAuth discovery for MCP connectors |
| GET/POST| `/oauth/authorize`            | admin  | consent screen (approved via admin login) |
| POST    | `/oauth/token`                | client pair | auth-code + PKCE, refresh |
| POST    | `/mcp`                        | oauth  | remote MCP server (Streamable HTTP) — see below |
| GET     | `/health`                     | open   | DB ping |

## Claude connector (MCP)

Conduct ships a remote MCP server so the Claude apps (iOS / desktop / web) can list and create jobs as a custom connector. It's an OAuth-protected Streamable-HTTP endpoint at `/mcp` exposing four tools: `list_task_types`, `list_jobs`, `get_job`, and `create_job` (asynchronous — create, then poll `get_job` for the result).

Conduct itself acts as the OAuth 2.0 authorization server (authorization-code + PKCE, refresh tokens). Each token binds to a client app, so MCP-created jobs inherit that client's attribution, rate limits, and cloud permissions. Deactivating a connector revokes all of its tokens.

Setup:

1. Set `CONDUCT_PUBLIC_URL` to your public origin (tunnel / reverse proxy) and `UI_COOKIE_SECURE=true` for HTTPS, then restart the API.
2. In the UI → **Connectors** → *New connector*: name it, bind it to a client app, and copy the generated Client ID + Secret (shown once).
3. In Claude → Settings → Connectors → *Add custom connector*: set the server URL to `https://<your-domain>/mcp`, then paste the Client ID and Secret under Advanced settings.
4. Connect → approve the consent screen in Conduct (admin login) → the four job tools appear in Claude.

## Read more

- **[docs/architecture.md](docs/architecture.md)** — sensitivity tiers, sync vs. async decision, routing engine, failure handling
- **[docs/deployment.md](docs/deployment.md)** — container build, host-side vs. containerized dev, git SHA provenance, private overlays, ECS / Cloud Run targets, ngrok
- **[docs/operations.md](docs/operations.md)** — live config, observability, common queries, tests, DoD
- **[docs/observability.md](docs/observability.md)** — wiring Conduct into the Watchtower LGTM stack (traces, logs, metrics, dashboard)
- **[docs/mcp-connector.md](docs/mcp-connector.md)** — the Claude custom connector: OAuth setup, minting a connector, connecting from the Claude apps
- **[docs/eval.md](docs/eval.md)** — comparing models for a task: shadows, review/scoring, the rollup, picking a model
- **[docs/prompts.md](docs/prompts.md)** — DB-backed prompts, resolution order, versioning, and the `conduct` CLI
- **[docs/auth.md](docs/auth.md)** — client API keys, admin auth, sensitivity tiers, and rate limits
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
oauth_provider.py          OAuth 2.0 authorization-server core (PKCE, tokens)
mcp_server.py              remote MCP server (FastMCP) for Claude connectors

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
scripts/cli.py             `conduct` admin CLI (prompts, jobs, routing)
tests/                     unit tests (pytest)
alembic/                   migrations
docs/                      verbose docs (architecture, deployment, operations)
Dockerfile                 multi-stage uv build; one image for api + worker
docker-compose.yml         postgres, redis, api, worker
```

---

© 2026 Zero Mission LLC
