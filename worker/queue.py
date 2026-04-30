"""RQ queue + worker entrypoint.

`make worker` runs `python -m worker.queue` which boots a SimpleWorker that
serves the `conduct` queue. SimpleWorker (vs Worker) doesn't fork — fits our
single-worker-single-job-at-a-time model and works on macOS without spawn
quirks.
"""

from __future__ import annotations

import logging
import sys

from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from prometheus_client import start_http_server as start_metrics_server
from redis import Redis
from rq import Queue, SimpleWorker

from config.settings import get_settings
from observability.tracing import init_tracing

QUEUE_NAME = "conduct"
DEFAULT_JOB_TIMEOUT_S = 600  # 10 minutes; long enough for cold-start swaps + inference
WORKER_METRICS_PORT = 8001

_redis: Redis | None = None
_queue: Queue | None = None


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


def queue_depth() -> int:
    """Pending jobs in the queue (does not include the in-flight job)."""
    return get_queue().count


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    init_tracing(
        service_name=settings.otel_service_name,
        otlp_endpoint=settings.otel_endpoint,
        role="worker",
    )
    HTTPXClientInstrumentor().instrument()
    # Prometheus pulls worker counters from a separate port. Alloy scrapes both.
    start_metrics_server(WORKER_METRICS_PORT)
    logging.getLogger(__name__).info("worker metrics on :%d/metrics", WORKER_METRICS_PORT)
    redis = get_redis()
    queue = Queue(QUEUE_NAME, connection=redis)
    worker = SimpleWorker([queue], connection=redis)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    sys.exit(main())
