# Local observability with Watchtower (LGTM)

Conduct is instrumented for the [Watchtower](https://github.com/gotoplanb/watchtower)
local LGTM stack (Loki, Grafana, Tempo, Prometheus, Alloy). This doc covers
wiring the two together for local development: traces, logs, metrics, and the
Conduct dashboard.

## How the signals flow

```
                         ┌────────── Tempo  (traces)
Conduct API ─┐  OTLP     │
             ├─ :4317 ─▶ Alloy ─────── Loki   (logs)
Conduct Wkr ─┘           │
                         └────────── Prometheus (metrics, via scrape)
                                        ▲
   API  :8000/metrics/prometheus ───────┘ (Alloy scrapes these
   Wkr  :8001/metrics            ────────┘  off the host)

                         Grafana :3000 reads Tempo + Loki + Prometheus
```

- **Traces and logs** are pushed over **OTLP gRPC** to Alloy (`:4317`).
- **Metrics** are `prometheus_client` counters/histograms exposed over HTTP and
  **scraped** by Alloy (they are not sent over OTLP).

## Prerequisites

- Watchtower running locally. Relevant endpoints: Grafana `:3000`, Prometheus
  `:9090`, Tempo `:3200`, Loki `:3100`, Alloy OTLP `:4317`.
- Conduct running (`make up`, or `make up-infra` + `make run` + `make worker`).

## 1. Traces + logs (OTLP — zero config)

Conduct exports both to `OTEL_EXPORTER_OTLP_ENDPOINT`:

- Containerized (`make up`): compose sets `host.docker.internal:4317` and maps
  the host gateway, so it reaches Alloy's published `:4317`.
- Host-side dev (`make run`): the default `localhost:4317` works directly.

What you get:

- **Traces → Tempo.** Service name `conduct`, with a `role` attribute of `api`
  or `worker`. Manual spans include `conduct.job` (root), `conduct.inference`,
  `conduct.worker.dispatch`, `conduct.worker.swap`, plus `conduct.tts`,
  `conduct.fanout`, and shadow spans.
- **Logs → Loki**, stamped with the active `trace_id`/`span_id`. In Grafana you
  can pivot from a log line straight to its trace in Tempo.

No setup beyond having Watchtower up — the OTLP exporter is initialized in the
API lifespan and the worker bootstrap.

## 2. Metrics (Alloy scrape — Watchtower side)

Conduct exposes open Prometheus endpoints:

- API: `:8000/metrics/prometheus`
- Worker: `:8001/metrics`

Alloy has to scrape these. To keep Conduct-specific targets **out of the public
Watchtower repo**, add them as a gitignored local override in your Watchtower
checkout rather than editing the committed Alloy config:

Create `docker/alloy-config.d/local-scrapes.alloy`:

```alloy
// Local Conduct scrape targets (gitignored — not committed to Watchtower).
prometheus.scrape "conduct_api" {
  targets         = [{ "__address__" = "host.docker.internal:8000", "job" = "conduct-api" }]
  metrics_path    = "/metrics/prometheus"
  scrape_interval = "15s"
  forward_to      = [prometheus.remote_write.local.receiver]
}

prometheus.scrape "conduct_worker" {
  targets         = [{ "__address__" = "host.docker.internal:8001", "job" = "conduct-worker" }]
  metrics_path    = "/metrics"
  scrape_interval = "15s"
  forward_to      = [prometheus.remote_write.local.receiver]
}
```

Then reload Alloy:

```bash
docker compose restart alloy   # in the Watchtower checkout
```

`prometheus.remote_write.local.receiver` is defined in Watchtower's main Alloy
config; the override just adds scrape jobs that feed it.

Verify the targets are up:

```bash
curl -s 'http://localhost:9090/api/v1/query?query=up{job=~"conduct.*"}' | jq '.data.result[] | {job: .metric.job, up: .value[1]}'
# expect conduct-api and conduct-worker both up=1
```

The metrics: `conduct_jobs_total`, `conduct_job_duration_seconds`,
`conduct_tokens_total`, `conduct_cost_usd_total`, `conduct_queue_depth`,
`conduct_model_swap_total`, `conduct_model_load_duration_seconds`.

## 3. The Conduct dashboard

Dashboards live in this repo at `grafana/dashboards/*.json` (versioned here, not
in the public Watchtower repo). Push them with:

```bash
make grafana-dashboards
```

This idempotently `POST`s each dashboard to Grafana by its `uid` (re-running
updates in place). Auth comes from the environment:

- `GRAFANA_URL` (default `http://localhost:3000`)
- `GRAFANA_TOKEN` (a service-account token), **or** `GRAFANA_USER` +
  `GRAFANA_PASSWORD`

The bundled **Conduct — Overview** dashboard (`/d/conduct-overview`) shows job
rate, failure rate, latency p50/p95, cost, token throughput, queue depth, and
model swaps — filterable by `task_type`.

## End-to-end smoke check

```bash
# metrics
curl -s 'http://localhost:9090/api/v1/query?query=up{job=~"conduct.*"}'
# logs
curl -s 'http://localhost:3100/loki/api/v1/label/job/values'   # includes conduct/conduct
# traces + dashboard
open http://localhost:3000                                     # Grafana
```

## Configuration reference

| Env var | Default | Purpose |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP target (traces + logs) → Alloy |
| `OTEL_SERVICE_NAME` | `conduct` | Service name in Tempo/Loki |
| `GRAFANA_BASE_URL` | `http://localhost:3000` | Used by the Conduct UI to deep-link a job to its trace in Grafana |
| `GRAFANA_URL` | `http://localhost:3000` | Target for `make grafana-dashboards` |
| `GRAFANA_TOKEN` / `GRAFANA_USER` + `GRAFANA_PASSWORD` | — | Auth for the dashboard push |

## Viewing traces

In Grafana, open **Explore → Tempo** and search for service `conduct` (or run
TraceQL like `{ .service.name = "conduct" }`). If a search looks empty, widen
the time range — Tempo search is time-window scoped, so a narrow recent window
can miss traces even though they're ingested. (An earlier "no traces in search"
report turned out to be exactly this, not a pipeline problem.) From the Conduct
UI, a job's detail page also deep-links straight to its trace.
```
