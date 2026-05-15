"""RQ queue + worker entrypoint.

`make worker` runs `python -m worker.queue` which boots a SimpleWorker that
serves the `conduct` queue. SimpleWorker (vs Worker) doesn't fork — fits our
single-worker-single-job-at-a-time model and works on macOS without spawn
quirks.
"""

from __future__ import annotations

import logging

from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from prometheus_client import start_http_server as start_metrics_server
from redis import Redis
from rq import Queue, SimpleWorker

from config.settings import get_settings
from observability.tracing import init_tracing

QUEUE_NAME = "conduct"
SHADOW_QUEUE_NAME = "conduct-shadow"
DEFAULT_JOB_TIMEOUT_S = 600  # 10 minutes; long enough for cold-start swaps + inference
WORKER_METRICS_PORT = 8001

_redis: Redis | None = None
_queue: Queue | None = None
_shadow_queue: Queue | None = None


def get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(get_settings().redis_url)
    return _redis


def get_queue() -> Queue:
    global _queue
    if _queue is None:
        _queue = Queue(QUEUE_NAME, connection=get_redis())
    return _queue


def get_shadow_queue() -> Queue:
    """Lower-priority queue for eval shadow jobs.

    SimpleWorker drains queues in list order, so by passing
    [main_queue, shadow_queue] the worker only pulls a shadow when the main
    queue is empty.
    """
    global _shadow_queue
    if _shadow_queue is None:
        _shadow_queue = Queue(SHADOW_QUEUE_NAME, connection=get_redis())
    return _shadow_queue


def queue_depth() -> int:
    """Pending jobs in the main queue (does not include the in-flight job)."""
    return get_queue().count


def shadow_queue_depth() -> int:
    return get_shadow_queue().count


def main() -> None:
    import asyncio

    from providers.resident import pin_resident_models, resident_model_names

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log = logging.getLogger(__name__)
    settings = get_settings()
    init_tracing(
        service_name=settings.otel_service_name,
        otlp_endpoint=settings.otel_endpoint,
        role="worker",
    )
    HTTPXClientInstrumentor().instrument()
    # Prometheus pulls worker counters from a separate port. Alloy scrapes both.
    start_metrics_server(WORKER_METRICS_PORT)
    log.info("worker metrics on :%d/metrics", WORKER_METRICS_PORT)

    # Pin resident models before accepting traffic so the API can fan out to
    # them immediately. Lazy import avoids a circular dep with worker.runner.
    if resident_model_names():
        from worker.runner import _get_providers

        providers = _get_providers()
        if providers.has("ollama"):
            ollama = providers.get("ollama")
            pinned = asyncio.run(pin_resident_models(ollama))
            log.info("pinned %d resident model(s): %s", len(pinned), pinned)

    redis = get_redis()
    main_queue = Queue(QUEUE_NAME, connection=redis)
    shadow_queue = Queue(SHADOW_QUEUE_NAME, connection=redis)
    # Order matters: SimpleWorker drains queues left-to-right, so shadows only
    # run when the main queue is empty.
    worker = SimpleWorker([main_queue, shadow_queue], connection=redis)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
