# Conduct

Multi-tenant LLM dispatch service. Routes AI workloads to local (Ollama) or cloud (Anthropic) models based on `task_type`, sensitivity, and availability. FastAPI + Postgres + Redis + RQ. See [SPEC.md](SPEC.md) for the design.

## Quickstart

```bash
make install                                                 # uv sync
make up                                                      # postgres + redis containers
make migrate                                                 # alembic upgrade head
cp .env.example .env                                         # fill in CONDUCT_ADMIN_KEY, ANTHROPIC_API_KEY
cp config/seed.clients.example.yaml config/seed.clients.yaml # then edit to add your clients
make seed                                                    # creates clients + 8 routing rules. Prints raw API keys ONCE.
make worker &                                                # RQ worker (handles all local-model jobs + model swaps)
make run                                                     # FastAPI on :8000
```

After `make seed`, save the printed client keys — they're hashed in the DB and unrecoverable.

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

Postgres + Redis are local to this repo's `docker-compose.yml`. Watchtower's LGTM stack (Tempo/Loki/Mimir/Grafana) lives in `~/watchtower` and receives traces + scrapes metrics.

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
| POST    | `/eval/jobs/{id}/score`       | admin  | manual quality rating 1–5 |
| GET     | `/health`                     | open   | DB ping |

## Sensitivity tiers

| tier            | routing                                      |
|-----------------|----------------------------------------------|
| `public`        | any model                                    |
| `internal`      | local preferred; cloud only when `ClientApp.allow_cloud_for_internal=true` |
| `confidential`  | local only — hard gate, no cloud fallback ever |

The routing rule's sensitivity acts as a floor — clients can be stricter, never looser.

## Sync vs async decision

- `"async": true` in body → enqueue, return 202
- Cloud target → sync, return 200 with full result
- Local target → enqueue (worker is the sole owner of Ollama + model swaps)

## Live config

- **Routing rules**: `PUT /routing/{task_type}` — DB-backed, read per request
- **Prompts**: edit files under `prompts/` — read fresh on every request, git hash captured per job
- **Pricing**: edit `config/pricing.yaml`, then `kill -USR1 $(pgrep -f '/.venv/bin/uvicorn main:app')`

## Private configuration (deployment-specific overrides)

Two things are deployment-specific and **must not be committed to a public fork**:

| Path | Status | Purpose |
|---|---|---|
| `config/seed.clients.yaml` | gitignored | client app names + per-client knobs (rate limits, cloud opt-in) |
| `prompts/clients/{client_name}/` | gitignored | task-prompt overrides specific to a deployment |

`config/seed.clients.example.yaml` and `prompts/clients/.example/` ship as templates. Two patterns for managing the real values:

- **Local files** (simplest): copy the examples into the gitignored paths and fill in.
- **Versioned via private repo** (recommended for orgs): keep your overrides in a separate private repo and mount as a submodule.
  ```bash
  git submodule add git@github.com:your-org/conduct-prompts.git prompts/clients
  git submodule update --init --recursive
  ```
  The gitignore rules already accommodate submodule contents — they won't be re-tracked by this repo.

`config/seed.routing.yaml` is committed because the starter rules are generic (task_types and model identifiers, no client identity). Override per-deployment by editing live (`PUT /routing/{task_type}`).

## Observability

- **Traces**: OTLP gRPC → `localhost:4317` (Watchtower's Alloy). Service name `conduct`, role `api` or `worker`
- **Prometheus**: API at `:8000/metrics/prometheus`, worker at `:8001/metrics`. Both exposed for Alloy scrape
- **Manual spans**: `conduct.job` (root, with task_type/sensitivity/client/model attrs), `conduct.inference` (child, with tokens/cost/latency), `conduct.worker.dispatch` and `conduct.worker.swap` for the worker leg

## Failure handling

`retry/static.py` is the v1 `FailureHandler` — on a `ProviderError`, returns `FALLBACK` if a distinct fallback model+provider is loaded, else `FAIL`. `retry/triage.py` is the v2 stub: implementing its `on_provider_error()` and binding it as the default in `worker/executor.py` is the only change needed to swap from heuristics to an LLM-driven decision. Tenacity retries Anthropic rate limits with exponential backoff (3 attempts, 1–10s).

## ngrok

```bash
echo "NGROK_AUTHTOKEN=..." >> .env
ngrok config add-authtoken "$NGROK_AUTHTOKEN"
make tunnel              # starts `ngrok http 8000`
```

Copy the printed HTTPS URL into the client config of whatever GCP service is calling Conduct. The tunnel is the only HTTPS surface; the underlying app speaks HTTP locally.

## Project layout

```
main.py                    FastAPI entrypoint + router registration
lifespan.py                tracing setup, providers, SIGUSR1 pricing reload
auth.py                    Bearer auth (client + admin)
deps.py                    shared deps (provider registry from app.state)
rate_limit.py              per-client Redis tumbling-window limiter
prompt_loader.py           clients/{name}/{task}.md → shared/{task}.md resolver

config/                    settings + pricing
db/                        SQLAlchemy 2.0 async session + declarative base
models/                    ORM models (ClientApp, ClientAppUsage, Job, RoutingRule) + Sensitivity/JobStatus enums
providers/                 BaseProvider, Ollama, Anthropic, registry
routing/                   pure decide() with sensitivity floor
worker/                    queue, runner (RQ entry), executor (sync+async share this)
retry/                     FailureHandler interface (static v1, triage v2 stub)
observability/             OTel tracer + Prometheus metric helpers
routes/                    route modules; one per concern
prompts/                   shared/ + clients/{name}/ overrides; .md files only
scripts/seed.py            idempotent bootstrap (reads config/seed.{clients,routing}.yaml)
tests/                     unit tests (pytest)
alembic/                   migrations
```

## Tests

```bash
make test                  # 39 unit tests, ~0.4s
```

Coverage: auth crypto, pricing registry + reload, prompt resolver hierarchy, both providers (mocked HTTP), routing engine (sensitivity floor, fallback stripping, defaults), queue decision + RQ round-trip via fakeredis, rate limit (under/over/separate clients), failure handler (fallback/fail conditions, triage stub).

End-to-end inference smoke (real Ollama + Anthropic) is documented but not in the test suite — it requires `ollama serve` running and an `ANTHROPIC_API_KEY`. See **DoD** below for the manual checks.

## Definition of Done — current state

Per [SPEC.md](SPEC.md):

- `POST /jobs` routes sync (cloud) and async (local + explicit). Sensitivity gates enforced.
- Worker processes queued jobs, performs model swaps, writes results.
- Model management routes (`GET /models`, `POST /models/{name}/load|unload`) operational.
- Routing config hot-reloadable via `PUT /routing/{task_type}` — DB-backed, no restart.
- `/metrics` and `/eval/compare` return aggregations with per-model attribution.
- A registered client app can submit a `bio_generation` job and get a response (with `ollama serve` running and a model pulled).
- ngrok wiring: `make tunnel` after setting `NGROK_AUTHTOKEN`.
- At least one client app created idempotently by `make seed` from `config/seed.clients.yaml`.
- 39 tests pass, lint + format clean.

## Common operations

```bash
# inspect the queue
make redis-cli
> LRANGE rq:queue:conduct 0 -1

# inspect jobs
make psql
> SELECT id, task_type, status, model_used FROM jobs ORDER BY created_at DESC LIMIT 10;

# rotate a client API key
make psql
> DELETE FROM client_apps WHERE name = 'foo';
> -- then POST /clients again to get a new key

# tail traces
open http://localhost:3000   # Grafana — Watchtower's instance
```
