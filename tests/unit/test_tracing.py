"""Trace-context propagation across the RQ hop (#49).

Uses a local TracerProvider (not the global one) so these tests don't fight
over the process-wide provider with other tests or leave it configured.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from observability.tracing import rq_trace_context, rq_trace_meta


def _local_tracer() -> tuple[trace.Tracer, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer(__name__), exporter


def test_rq_trace_meta_carries_active_span() -> None:
    tracer, _ = _local_tracer()
    with tracer.start_as_current_span("enqueue") as span:
        meta = rq_trace_meta()
    trace_id = span.get_span_context().trace_id
    assert f"{trace_id:032x}" in meta["traceparent"]


def test_rq_trace_meta_empty_without_active_span() -> None:
    assert rq_trace_meta() == {}


def test_rq_trace_context_round_trip_parents_worker_span() -> None:
    tracer, exporter = _local_tracer()
    with tracer.start_as_current_span("api.request") as api_span:
        meta = rq_trace_meta()

    # Simulate the worker process: no ambient span, meta from the RQ job.
    with rq_trace_context(meta), tracer.start_as_current_span("conduct.job"):
        pass

    api_ctx = api_span.get_span_context()
    worker_span = next(s for s in exporter.get_finished_spans() if s.name == "conduct.job")
    assert worker_span.context.trace_id == api_ctx.trace_id
    assert worker_span.parent is not None
    assert worker_span.parent.span_id == api_ctx.span_id


def test_rq_trace_context_noop_on_empty_meta() -> None:
    tracer, exporter = _local_tracer()
    with rq_trace_context({}), tracer.start_as_current_span("conduct.job"):
        pass
    (span,) = exporter.get_finished_spans()
    assert span.parent is None


def test_rq_trace_context_noop_outside_rq() -> None:
    # meta=None falls back to rq.get_current_job(), which is None outside a
    # worker — must not raise and must not attach anything.
    with rq_trace_context():
        assert not trace.get_current_span().get_span_context().is_valid
