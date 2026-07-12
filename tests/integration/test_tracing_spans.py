"""Regression for #49: the API must emit server spans.

instrument_app used to run inside the lifespan body — after Starlette froze
the middleware stack — so the OTel middleware never entered the request path
and the API produced zero spans. Instrumentation now happens at import time
in main.py; this test sets a real tracer provider (the proxy tracer picks it
up, exactly as init_tracing does in production) and asserts a request
actually yields a server span that honors an incoming traceparent.
"""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture(scope="module")
def span_exporter() -> InMemorySpanExporter:
    """Set the process-global tracer provider once, with an in-memory sink.

    set_tracer_provider only takes effect on the first call per process; if
    another test set a bare provider first, spans would go nowhere and this
    test would fail — which is the correct signal, since production likewise
    depends on the proxy tracer resolving to the real provider.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


async def test_health_request_produces_server_span(client, span_exporter) -> None:
    span_exporter.clear()
    resp = await client.get("/health")
    assert resp.status_code == 200
    names = [s.name for s in span_exporter.get_finished_spans()]
    assert "GET /health" in names


async def test_server_span_continues_incoming_traceparent(client, span_exporter) -> None:
    span_exporter.clear()
    trace_id = "af7651916cd43dd8448eb211c80319c6"
    parent_span_id = "b7ad6b7169203331"
    resp = await client.get(
        "/health",
        headers={"traceparent": f"00-{trace_id}-{parent_span_id}-01"},
    )
    assert resp.status_code == 200
    server_span = next(
        s for s in span_exporter.get_finished_spans() if s.name == "GET /health"
    )
    assert f"{server_span.context.trace_id:032x}" == trace_id
    assert server_span.parent is not None
    assert f"{server_span.parent.span_id:016x}" == parent_span_id
