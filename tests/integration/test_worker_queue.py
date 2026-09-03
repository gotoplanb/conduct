"""Worker boot wiring (worker/queue.py).

main() is the process entrypoint — the tests pin the wiring contract (queue
priority order, scheduler off, metrics port, resident pinning) with the
side-effectful collaborators stubbed. The lazy queue singletons run against
fakeredis; the reconcile loop and SIGUSR1 handler are driven deterministically
(fake sleep, direct handler invocation) — no real waits, no real signals.
"""

from __future__ import annotations

import logging
import threading
from uuid import uuid4

import fakeredis
import pytest

import worker.queue as wq

# --- lazy singletons ---


@pytest.fixture
def fresh_queue_module(monkeypatch):
    """Reset the module-global singletons and point Redis.from_url at a fake
    so get_redis/get_queue construct against it."""
    fake = fakeredis.FakeRedis()

    class _Factory:
        @staticmethod
        def from_url(url: str) -> fakeredis.FakeRedis:
            _Factory.seen_url = url
            return fake

    monkeypatch.setattr(wq, "Redis", _Factory)
    monkeypatch.setattr(wq, "_redis", None)
    monkeypatch.setattr(wq, "_queue", None)
    monkeypatch.setattr(wq, "_media_queue", None)
    monkeypatch.setattr(wq, "_shadow_queue", None)
    return fake


def test_get_redis_builds_once_from_settings_url(fresh_queue_module) -> None:
    r = wq.get_redis()
    assert r is fresh_queue_module
    assert wq.Redis.seen_url  # constructed from settings.redis_url
    assert wq.get_redis() is r  # cached, not rebuilt


def test_queue_singletons_names_and_caching(fresh_queue_module) -> None:
    q, mq, sq = wq.get_queue(), wq.get_media_queue(), wq.get_shadow_queue()
    assert (q.name, mq.name, sq.name) == ("conduct", "conduct-media", "conduct-shadow")
    assert wq.get_queue() is q
    assert wq.get_media_queue() is mq
    assert wq.get_shadow_queue() is sq
    # All three share the one connection.
    assert q.connection is fresh_queue_module


def test_queue_depths_read_from_redis(fresh_queue_module) -> None:
    assert wq.queue_depth() == 0
    assert wq.shadow_queue_depth() == 0
    wq.get_queue().enqueue_call(func=print, args=(1,), job_id=str(uuid4()))
    assert wq.queue_depth() == 1
    assert wq.shadow_queue_depth() == 0  # main enqueue doesn't leak into shadows


# --- main() wiring ---


@pytest.fixture
def main_stubs(fresh_queue_module, monkeypatch):
    """Neutralize main()'s side-effectful collaborators; record what it wires."""
    import providers.resident as resident

    calls: dict = {}
    monkeypatch.setattr(
        wq, "init_tracing", lambda **kwargs: calls.setdefault("tracing", kwargs)
    )
    monkeypatch.setattr(
        wq, "init_logging", lambda **kwargs: calls.setdefault("logging", kwargs)
    )

    class _Instrumentor:
        def instrument(self) -> None:
            calls["httpx_instrumented"] = True

    monkeypatch.setattr(wq, "HTTPXClientInstrumentor", _Instrumentor)
    monkeypatch.setattr(
        wq, "start_metrics_server", lambda port: calls.setdefault("metrics_port", port)
    )
    monkeypatch.setattr(
        wq, "_install_sigusr1_pricing_reload",
        lambda log: calls.setdefault("sigusr1_installed", True),
    )
    monkeypatch.setattr(resident, "resident_model_names", lambda: [])

    class _FakeWorker:
        def __init__(self, queues, connection=None):
            calls["queue_order"] = [q.name for q in queues]
            calls["connection"] = connection

        def work(self, with_scheduler=None) -> None:
            calls["with_scheduler"] = with_scheduler

    monkeypatch.setattr(wq, "SimpleWorker", _FakeWorker)
    return calls


def test_main_wires_worker_and_priority_order(main_stubs) -> None:
    wq.main()
    # Order is the priority contract: text > media > shadows.
    assert main_stubs["queue_order"] == ["conduct", "conduct-media", "conduct-shadow"]
    assert main_stubs["with_scheduler"] is False
    assert main_stubs["metrics_port"] == wq.WORKER_METRICS_PORT
    assert main_stubs["httpx_instrumented"] is True
    assert main_stubs["sigusr1_installed"] is True
    assert main_stubs["tracing"]["role"] == "worker"
    assert main_stubs["logging"]["role"] == "worker"


def test_main_pins_residents_before_serving(main_stubs, monkeypatch) -> None:
    import providers.resident as resident
    import worker.runner as runner

    class _Ollama:
        name = "ollama"

    class _Registry:
        def has(self, name: str) -> bool:
            return name == "ollama"

        def get(self, name: str) -> _Ollama:
            return _Ollama()

    pinned = {}

    async def fake_pin(ollama):
        pinned["ollama"] = ollama
        return ["gemma4:e4b"]

    monkeypatch.setattr(resident, "resident_model_names", lambda: ["gemma4:e4b"])
    monkeypatch.setattr(resident, "pin_resident_models", fake_pin)
    monkeypatch.setattr(runner, "_get_providers", lambda: _Registry())
    monkeypatch.setattr(
        wq, "_start_resident_reconcile",
        lambda ollama, interval, log: pinned.setdefault("reconcile_interval", interval),
    )

    wq.main()
    assert isinstance(pinned["ollama"], _Ollama)
    # The self-heal loop is armed with the configured interval (#47).
    assert pinned["reconcile_interval"] == wq.get_settings().resident_reconcile_interval_s
    # Pinning happens before the worker starts draining.
    assert main_stubs["with_scheduler"] is False


# --- _start_resident_reconcile ---


def test_reconcile_disabled_when_interval_nonpositive(caplog) -> None:
    with caplog.at_level(logging.INFO):
        wq._start_resident_reconcile(object(), 0, logging.getLogger("test-reconcile"))
    assert any("resident reconcile disabled" in r.message for r in caplog.records)
    assert not any(t.name == "resident-reconcile" for t in threading.enumerate())


# The loop has no exit path by design (daemon thread dies with the process),
# so the test terminates it by raising out of the fake sleep — pytest's
# thread-exception check would flag that escape, hence the filter.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_reconcile_loop_survives_failure_and_repins(monkeypatch, caplog) -> None:
    # Drive the daemon loop deterministically: fake time.sleep so tick 1
    # raises inside reconcile (must not kill the loop), tick 2 re-pins, and
    # the third sleep exits the thread via SystemExit.
    import time

    import providers.resident as resident

    ticks: list[object] = []
    done = threading.Event()

    def fake_sleep(_interval: float) -> None:
        if len(ticks) >= 2:
            done.set()
            raise SystemExit

    async def fake_reconcile(ollama):
        ticks.append(ollama)
        if len(ticks) == 1:
            raise RuntimeError("ps failed")
        return ["gemma4:e4b"]

    monkeypatch.setattr(time, "sleep", fake_sleep)
    monkeypatch.setattr(resident, "reconcile_resident_models", fake_reconcile)

    sentinel = object()
    with caplog.at_level(logging.INFO, logger="test-reconcile"):
        wq._start_resident_reconcile(sentinel, 30, logging.getLogger("test-reconcile"))
        assert done.wait(timeout=5.0)
        thread = next(t for t in threading.enumerate() if t.name == "resident-reconcile")
        thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert ticks == [sentinel, sentinel]
    messages = [r.message for r in caplog.records]
    assert any("resident reconcile tick failed" in m for m in messages)
    assert any("re-pinned" in m for m in messages)


# --- SIGUSR1 pricing reload ---


@pytest.fixture
def restore_sigusr1():
    import signal

    old = signal.getsignal(signal.SIGUSR1)
    yield signal
    signal.signal(signal.SIGUSR1, old)


def test_sigusr1_handler_reloads_pricing(restore_sigusr1, monkeypatch, caplog) -> None:
    import config.pricing as pricing_mod

    signal = restore_sigusr1
    reloads: list[bool] = []

    class _Pricing:
        path = "/etc/conduct/pricing.yaml"

        def reload(self) -> None:
            reloads.append(True)

    monkeypatch.setattr(pricing_mod, "get_pricing", lambda: _Pricing())
    with caplog.at_level(logging.INFO, logger="test-sigusr1"):
        wq._install_sigusr1_pricing_reload(logging.getLogger("test-sigusr1"))
        handler = signal.getsignal(signal.SIGUSR1)
        handler(signal.SIGUSR1, None)  # invoke directly — no real signal delivery
    assert reloads == [True]
    assert any("pricing reloaded" in r.message for r in caplog.records)


def test_sigusr1_handler_survives_reload_failure(
    restore_sigusr1, monkeypatch, caplog
) -> None:
    import config.pricing as pricing_mod

    signal = restore_sigusr1

    class _BrokenPricing:
        path = "/etc/conduct/pricing.yaml"

        def reload(self) -> None:
            raise RuntimeError("yaml parse error")

    monkeypatch.setattr(pricing_mod, "get_pricing", lambda: _BrokenPricing())
    with caplog.at_level(logging.INFO, logger="test-sigusr1"):
        wq._install_sigusr1_pricing_reload(logging.getLogger("test-sigusr1"))
        handler = signal.getsignal(signal.SIGUSR1)
        handler(signal.SIGUSR1, None)  # a bad pricing.yaml must not kill the worker
    assert any("pricing reload failed" in r.message for r in caplog.records)


def test_sigusr1_install_degrades_off_main_thread(caplog) -> None:
    # signal.signal only works on the main thread — the worker logs and moves
    # on rather than crashing (the Windows / embedded-test path).
    with caplog.at_level(logging.INFO, logger="test-sigusr1"):
        t = threading.Thread(
            target=wq._install_sigusr1_pricing_reload,
            args=(logging.getLogger("test-sigusr1"),),
        )
        t.start()
        t.join(timeout=5.0)
    assert any("SIGUSR1 not available" in r.message for r in caplog.records)
