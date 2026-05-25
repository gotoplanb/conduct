# Operations

Day-to-day running: live config, observability, common queries, tests.

## Live config

Three things can change at runtime without a restart:

- **Routing rules** — `PUT /routing/{task_type}` writes to the DB and is read on every job. Hot-swap in real time.
- **Prompts** — stored in the Postgres `prompts` table; edit with the `conduct` CLI or `PUT /prompts/{task_type}`. Every save appends a row to `prompt_versions`, and each job records the resolved `version_id` in `Job.metadata.prompt` so the exact content a job ran against can be replayed after later edits.
- **Pricing** — edit `config/pricing.yaml`, then signal the API:
  ```bash
  kill -USR1 $(pgrep -f '/.venv/bin/uvicorn main:app')
  ```
  We use `SIGUSR1` rather than `SIGHUP` because uvicorn intercepts `SIGHUP` for shutdown.

### Editing prompts

`prompts/shared/*.md` and `prompts/clients/*/*.md` are import-time seed material — `scripts/seed.py` walks them on first run and inserts each file as a `prompts` row. After that, they're advisory; the DB is the source of truth. Round-trip through the CLI:

```bash
export CONDUCT_ADMIN_KEY=...                # required
export CONDUCT_BASE_URL=http://localhost:8000  # optional; this is the default

conduct prompts list                         # all prompts, shared + client-specific
conduct prompts get bio_generation           # print current content
conduct prompts get bio_generation --client bosshardt-portal
conduct prompts edit bio_generation          # opens $EDITOR; PUTs on save
conduct prompts history bio_generation       # version log
```

`edit` works like `git commit -e`: it pulls the current row, drops it in `$EDITOR` (falls back to `vi`), and on save sends a `PUT` to `/prompts/{task_type}` — which both updates the live row and appends a `prompt_versions` entry. Saving an empty or unchanged buffer aborts cleanly.

## Observability

The Watchtower LGTM stack (Tempo / Loki / Prometheus / Grafana / Alloy) runs as a separate Docker Compose project. Conduct sends traces **and logs** to it over OTLP and exposes Prometheus endpoints for Alloy to scrape. For the full setup walkthrough — OTLP wiring, the gitignored Alloy scrape override, and pushing the Conduct dashboard — see **[observability.md](observability.md)**.

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

### Logs

Python logs are exported over OTLP to Alloy → Loki, stamped with the active `trace_id`/`span_id` so you can pivot from a log line to its trace in Grafana.

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
- Containerized via `make up`. Prompt provenance survives via `prompt_versions` rows in the DB — each job records the `version_id` it resolved to.
