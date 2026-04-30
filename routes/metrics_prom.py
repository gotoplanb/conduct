"""Prometheus scrape endpoint. No auth — Alloy needs to reach it."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

from observability.metrics import QUEUE_DEPTH
from worker.queue import queue_depth

log = logging.getLogger(__name__)

router = APIRouter(tags=["observability"])


@router.get("/metrics/prometheus", include_in_schema=False)
async def metrics_prom() -> Response:
    # Refresh the queue gauge at scrape time so it's accurate.
    try:
        QUEUE_DEPTH.labels(priority="all").set(queue_depth())
    except Exception as e:  # Redis hiccup — emit 0 rather than 5xx
        log.warning("queue depth scrape failed: %s", e)
        QUEUE_DEPTH.labels(priority="all").set(0)
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
