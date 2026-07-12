"""OpenTelemetry tracer setup. Pushes OTLP gRPC to Watchtower's Alloy."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

log = logging.getLogger(__name__)

_initialized = False


def init_tracing(*, service_name: str, otlp_endpoint: str, role: str = "api") -> None:
    """Idempotent tracer setup. Call once per process at startup.

    `role` distinguishes the API process from the worker in Tempo (e.g. via the
    `service.namespace` resource attr).
    """
    global _initialized
    if _initialized:
        return

    resource = Resource.create(
        {
            SERVICE_NAME: service_name,
            "service.namespace": "conduct",
            "service.instance.role": role,
        }
    )
    provider = TracerProvider(resource=resource)
    try:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    except Exception as e:  # pragma: no cover - exporter init shouldn't normally fail
        log.warning("OTLP exporter setup failed (%s) — traces will not export", e)

    trace.set_tracer_provider(provider)
    _initialized = True
    log.info(
        "OTel tracing initialized: service=%s role=%s endpoint=%s pid=%s",
        service_name,
        role,
        otlp_endpoint,
        os.getpid(),
    )


def get_tracer(name: str = "conduct"):
    return trace.get_tracer(name)


# W3C trace-context keys worth carrying across the RQ hop (#49). RQ meta is
# pickled into Redis, so we stash only these two strings — never live objects.
_CARRIER_KEYS = ("traceparent", "tracestate")


def rq_trace_meta() -> dict[str, str]:
    """Trace-context carrier to pass as `meta=` when enqueuing an RQ job.

    Without this the worker's spans start as root spans in their own traces,
    disconnected from the request that enqueued them (#49). Empty dict when
    there is no active span — RQ treats meta={} as no meta.
    """
    carrier: dict[str, str] = {}
    inject(carrier)
    return {k: carrier[k] for k in _CARRIER_KEYS if k in carrier}


@contextmanager
def rq_trace_context(meta: Mapping[str, str] | None = None) -> Iterator[None]:
    """Continue the enqueuing request's trace inside an RQ job function.

    Reads the carrier stashed by `rq_trace_meta` (from the current RQ job's
    meta unless one is passed explicitly) and attaches it as the ambient
    context, so every span opened inside — including those in tasks created
    by asyncio.run, which copies contextvars — parents onto the API's trace.
    No-op when there is nothing to extract.
    """
    if meta is None:
        from rq import get_current_job  # noqa: PLC0415 — worker-only dep path

        rq_job = get_current_job()
        meta = rq_job.meta if rq_job is not None and rq_job.meta else {}
    carrier = {k: meta[k] for k in _CARRIER_KEYS if meta.get(k)}
    token = otel_context.attach(extract(carrier)) if carrier else None
    try:
        yield
    finally:
        if token is not None:
            otel_context.detach(token)
