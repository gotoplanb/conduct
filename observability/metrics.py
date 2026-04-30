"""Prometheus metric definitions and recording helpers.

All metrics live in the global `prometheus_client` REGISTRY. The
`/metrics/prometheus` route serializes the registry on scrape.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

JOBS_TOTAL = Counter(
    "conduct_jobs_total",
    "Jobs by terminal status, task_type, model, and client_app",
    ["status", "task_type", "model", "client_app"],
)

JOB_DURATION_SECONDS = Histogram(
    "conduct_job_duration_seconds",
    "End-to-end job duration in seconds",
    ["task_type", "model"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600),
)

TOKENS_TOTAL = Counter(
    "conduct_tokens_total",
    "Cumulative tokens by direction (in/out), model, and client_app",
    ["direction", "model", "client_app"],
)

COST_USD_TOTAL = Counter(
    "conduct_cost_usd_total",
    "Cumulative USD cost by model and client_app",
    ["model", "client_app"],
)

MODEL_LOAD_DURATION_SECONDS = Histogram(
    "conduct_model_load_duration_seconds",
    "Time spent loading a model into Ollama memory",
    ["model"],
    buckets=(1, 5, 10, 30, 60, 120, 300),
)

MODEL_SWAP_TOTAL = Counter(
    "conduct_model_swap_total",
    "Number of model swaps performed by the worker",
    ["from_model", "to_model"],
)

QUEUE_DEPTH = Gauge(
    "conduct_queue_depth",
    "Pending jobs in the RQ queue (set on Prometheus scrape)",
    ["priority"],
)

RETRY_TOTAL = Counter(
    "conduct_retry_total",
    "Provider retries (incremented by the static retry layer in P9)",
    ["provider", "reason"],
)

FALLBACK_TOTAL = Counter(
    "conduct_fallback_total",
    "Fallback invocations (primary failed, fallback attempted)",
    ["from_provider", "to_provider", "reason"],
)


def record_job_completion(
    *,
    status: str,
    task_type: str,
    model: str,
    client_app: str,
    duration_s: float | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
) -> None:
    JOBS_TOTAL.labels(
        status=status,
        task_type=task_type,
        model=model or "<none>",
        client_app=client_app,
    ).inc()
    if duration_s is not None and (model or "") != "":
        JOB_DURATION_SECONDS.labels(task_type=task_type, model=model).observe(duration_s)
    if tokens_in:
        TOKENS_TOTAL.labels(direction="in", model=model, client_app=client_app).inc(tokens_in)
    if tokens_out:
        TOKENS_TOTAL.labels(direction="out", model=model, client_app=client_app).inc(tokens_out)
    if cost_usd:
        COST_USD_TOTAL.labels(model=model, client_app=client_app).inc(cost_usd)


def record_model_swap(*, from_model: str, to_model: str, duration_s: float) -> None:
    MODEL_SWAP_TOTAL.labels(from_model=from_model or "<none>", to_model=to_model).inc()
    MODEL_LOAD_DURATION_SECONDS.labels(model=to_model).observe(duration_s)


def record_fallback(*, from_provider: str, to_provider: str, reason: str) -> None:
    FALLBACK_TOTAL.labels(from_provider=from_provider, to_provider=to_provider, reason=reason).inc()


def record_retry(*, provider: str, reason: str) -> None:
    RETRY_TOTAL.labels(provider=provider, reason=reason).inc()
