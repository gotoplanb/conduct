"""OpenTelemetry tracer setup. Pushes OTLP gRPC to Watchtower's Alloy."""

from __future__ import annotations

import logging
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
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
