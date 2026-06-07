"""Worker SIGUSR1 → pricing reload (mirror of the API's lifespan handler).

The worker is the process that prices shadow jobs, so without its own
signal handler an operator edit + `make reload-pricing` would silently
leave the worker on stale rates. These tests pin the install path:
the handler must register against SIGUSR1, and firing the signal must
call PricingRegistry.reload().
"""

from __future__ import annotations

import logging
import signal
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fresh_pricing(monkeypatch, tmp_path):
    """Reset config.pricing._registry around each test so signal-handler
    calls actually hit reload() rather than picking up a stub from earlier
    suites."""
    import config.pricing as pricing_mod

    p = tmp_path / "pricing.yaml"
    p.write_text(
        "anthropic:\n"
        "  claude-haiku-4-5:\n"
        "    input_per_1m_usd: 1.0\n"
        "    output_per_1m_usd: 5.0\n"
    )
    monkeypatch.setattr(pricing_mod, "_registry", None)
    monkeypatch.setattr(pricing_mod, "DEFAULT_PRICING_PATH", p)
    yield p


def test_install_sigusr1_registers_handler(fresh_pricing) -> None:
    """The install helper must put a SIGUSR1 handler in place. Importing the
    worker module shouldn't auto-register — only the main() path should."""
    from worker.queue import _install_sigusr1_pricing_reload

    prior = signal.getsignal(signal.SIGUSR1)
    try:
        _install_sigusr1_pricing_reload(logging.getLogger("test"))
        installed = signal.getsignal(signal.SIGUSR1)
        assert installed is not prior
        assert callable(installed)
    finally:
        signal.signal(signal.SIGUSR1, prior or signal.SIG_DFL)


def test_sigusr1_handler_calls_pricing_reload(fresh_pricing) -> None:
    """Invoke the registered handler directly with stub args and confirm it
    calls PricingRegistry.reload() — not relying on actually delivering a
    real signal, which is fragile under pytest."""
    from worker.queue import _install_sigusr1_pricing_reload

    prior = signal.getsignal(signal.SIGUSR1)
    try:
        with patch("config.pricing.PricingRegistry.reload") as mock_reload:
            _install_sigusr1_pricing_reload(logging.getLogger("test"))
            handler = signal.getsignal(signal.SIGUSR1)
            assert callable(handler)
            # signum + frame are unused by our handler.
            handler(signal.SIGUSR1, None)
            assert mock_reload.called
    finally:
        signal.signal(signal.SIGUSR1, prior or signal.SIG_DFL)


def test_sigusr1_handler_swallows_reload_errors(fresh_pricing, caplog) -> None:
    """A bad pricing.yaml shouldn't crash the worker — the handler logs and
    moves on. Without this guarantee, an operator edit could leave the
    worker dead instead of just stale."""
    from worker.queue import _install_sigusr1_pricing_reload

    prior = signal.getsignal(signal.SIGUSR1)
    try:
        with patch("config.pricing.PricingRegistry.reload", side_effect=RuntimeError("boom")):
            _install_sigusr1_pricing_reload(logging.getLogger("test"))
            handler = signal.getsignal(signal.SIGUSR1)
            with caplog.at_level(logging.ERROR):
                handler(signal.SIGUSR1, None)
            assert any("pricing reload failed" in r.message for r in caplog.records)
    finally:
        signal.signal(signal.SIGUSR1, prior or signal.SIG_DFL)


def test_install_skips_gracefully_when_signal_not_available(monkeypatch) -> None:
    """On platforms without SIGUSR1 (Windows) or when signal.signal raises,
    the install must log and continue rather than crash worker startup."""
    from worker.queue import _install_sigusr1_pricing_reload

    def _raise(*_a, **_kw):
        raise ValueError("signal only works in main thread")

    monkeypatch.setattr(signal, "signal", _raise)
    log = MagicMock()
    # Should not raise.
    _install_sigusr1_pricing_reload(log)
    # And should have noted the absence at INFO level.
    assert log.info.called
