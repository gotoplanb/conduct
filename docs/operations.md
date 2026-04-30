# Operations

Day-to-day running: live config, observability, common queries, tests.

## Live config

Three things can change at runtime without a restart:

- **Routing rules** — `PUT /routing/{task_type}` writes to the DB and is read on every job. Hot-swap in real time.
- **Prompts** — `prompts/shared/*.md` and `prompts/clients/*/*.md` are read fresh on every request. Save the file → next job uses the new version. Each job records the resolved file path + git hash in `Job.metadata.prompt` for auditability.
- **Pricing** — edit `config/pricing.yaml`, then signal the API:
  ```bash
  kill -USR1 $(pgrep -f '/.venv/bin/uvicorn main:app')
  ```
  We use `SIGUSR1` rather than `SIGHUP` because uvicorn intercepts `SIGHUP` for shutdown.

In a containerized run, `prompts/` is baked into the image — edits to local files won't show up until rebuild. That's a known limitation; see [deployment.md](deployment.md#cloud-target-ecs--google-cloud-run) for the unsolved cloud-prompt-storage question.

## Observability

The Watchtower LGTM stack (Tempo / Loki / Mimir / Grafana / Alloy) lives at `~/watchtower` and runs as a separate Docker Compose project. Conduct sends traces to it and exposes Prometheus endpoints for it to scrape.

### Traces (OpenTelemetry)

- Exporter: OTLP gRPC → `localhost:4317` (host-side dev) or `host.docker.internal:4317` (containerized).
- Service name: `conduct`. Role attribute: `api` or `worker`.
- Manual spans:
  - `conduct.job` — root span. Attrs: `task_type`, `sensitivity`, `client`, `model`.
  - `conduct.inference` — child of `conduct.job`. Attrs: `tokens_in`, `tokens_out`, `cost_usd`, `latency_ms`.
  - `conduct.worker.dispatch` — worker leg of an async job.
  - `conduct.worker.swap` — Ollama model swap, separate so cold-start cost is visible.

### Prometheus

- API: `:8000/metrics/prometheus` (open endpoint, scrape target).
- Worker: `:8001/metrics` (open endpoint, scrape target).

Counters and histograms cover job count, latency, tokens, cost, and queue depth. Both are scraped by Watchtower's Alloy.

## Failure handling

See [architecture.md#failure-handling](architecture.md#failure-handling) for the FailureHandler interface (Static v1, Triage v2 stub).

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
> -- (or re-run `make seed` if 'foo' is in seed.clients.yaml)

# tail traces
open http://localhost:3000   # Grafana — Watchtower's instance
```

## Tests

```bash
make test                  # 41 unit tests, ~0.4s
make lint                  # ruff check
make format                # ruff format
```

Coverage:

- Auth crypto (key generation, hashing, comparison)
- Pricing registry + SIGUSR1 reload
- Prompt resolver hierarchy + git-hash fallback (host + containerized paths)
- Both providers (mocked HTTP via respx)
- Routing engine — sensitivity floor, fallback stripping, defaults
- Queue decision + RQ round-trip via fakeredis
- Rate limit — under/over/separate clients
- Failure handler — fallback / fail conditions, triage stub

End-to-end inference (real Ollama + Anthropic) is documented but not in the suite — it requires `ollama serve` running and an `ANTHROPIC_API_KEY`. See **DoD** below for manual checks.

## Definition of Done — current state

Per [SPEC.md](../SPEC.md):

- `POST /jobs` routes sync (cloud) and async (local + explicit). Sensitivity gates enforced.
- Worker processes queued jobs, performs model swaps, writes results.
- Model management routes (`GET /models`, `POST /models/{name}/load|unload`) operational.
- Routing config hot-reloadable via `PUT /routing/{task_type}` — DB-backed, no restart.
- `/metrics` and `/eval/compare` return aggregations with per-model attribution.
- A registered client app can submit a `bio_generation` job and get a response (with `ollama serve` running and a model pulled).
- ngrok wiring: `make tunnel` after setting `NGROK_AUTHTOKEN`.
- At least one client app created idempotently by `make seed` from `config/seed.clients.yaml`.
- 41 tests pass, lint + format clean.
- Containerized via `make up`; image carries baked-in `CONDUCT_GIT_SHA` for prompt provenance.
