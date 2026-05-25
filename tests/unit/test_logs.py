"""Unit test for the OTLP logging init — handler attach + idempotency,
without standing up a real gRPC exporter."""

from __future__ import annotations

import logging

from observability import logs


class _DummyExporter:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def export(self, *args, **kwargs):
        return None

    def shutdown(self, *args, **kwargs) -> None:
        return None

    def force_flush(self, *args, **kwargs) -> bool:
        return True


def test_init_logging_attaches_handler_once(monkeypatch) -> None:
    monkeypatch.setattr(logs, "OTLPLogExporter", _DummyExporter)
    monkeypatch.setattr(logs, "_initialized", False)
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        logs.init_logging(service_name="conduct", otlp_endpoint="http://x:4317", role="test")
        assert logs._initialized is True
        added = [h for h in root.handlers if h not in before]
        assert len(added) == 1
        assert isinstance(added[0], logs.LoggingHandler)

        # Second call is a no-op (no duplicate handler).
        logs.init_logging(service_name="conduct", otlp_endpoint="http://x:4317", role="test")
        assert len([h for h in root.handlers if h not in before]) == 1
    finally:
        root.handlers = before
        monkeypatch.setattr(logs, "_initialized", False)
