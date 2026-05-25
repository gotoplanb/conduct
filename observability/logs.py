"""OpenTelemetry log export. Ships Python logging records to Watchtower's
Alloy over OTLP (same endpoint as traces), which forwards them to Loki.

The OTel LoggingHandler stamps each record with the active span's trace_id /
span_id, so a log line in Loki links straight to its trace in Tempo — the
main payoff of routing logs through OTel rather than scraping stdout.
"""

from __future__ import annotations

import logging

from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

log = logging.getLogger(__name__)

_initialized = False


def init_logging(
    *, service_name: str, otlp_endpoint: str, role: str = "api", level: int = logging.INFO
) -> None:
    """Idempotent. Attaches an OTLP log handler to the root logger so all
    app logs are exported, in addition to the existing stdout handler."""
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
    provider = LoggerProvider(resource=resource)
    try:
        provider.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter(endpoint=otlp_endpoint, insecure=True))
        )
    except Exception as e:  # pragma: no cover - exporter init shouldn't normally fail
        log.warning("OTLP log exporter setup failed (%s) — logs will not export", e)
        return

    set_logger_provider(provider)
    handler = LoggingHandler(level=level, logger_provider=provider)
    logging.getLogger().addHandler(handler)
    _initialized = True
