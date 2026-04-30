"""Sync/async decision + RQ enqueue mechanics with fakeredis.

These tests don't run the worker — they verify what the API would put on the
queue and how `_should_enqueue` classifies decisions.
"""

from __future__ import annotations

import fakeredis
import pytest
from rq import Queue

from models.types import Sensitivity
from routes.jobs import JobCreateIn, _should_enqueue
from routing.engine import RoutingDecision


def _decision(provider: str, model: str = "x") -> RoutingDecision:
    return RoutingDecision(
        model=model,
        provider=provider,
        fallback_model=None,
        fallback_provider=None,
        effective_sensitivity=Sensitivity.PUBLIC,
        max_tokens=1000,
        reason="test",
    )


def _body(**overrides) -> JobCreateIn:
    base = {"task_type": "x", "prompt": "p"}
    return JobCreateIn(**(base | overrides))


def test_async_flag_forces_enqueue_even_for_cloud() -> None:
    assert _should_enqueue(_body(**{"async": True}), _decision("anthropic")) is True


def test_local_target_always_enqueued() -> None:
    assert _should_enqueue(_body(), _decision("ollama")) is True


def test_cloud_target_runs_sync_by_default() -> None:
    assert _should_enqueue(_body(), _decision("anthropic")) is False


@pytest.fixture
def fake_queue() -> Queue:
    fake = fakeredis.FakeStrictRedis()
    return Queue("conduct-test", connection=fake)


def test_enqueue_with_job_id_is_fetchable(fake_queue: Queue) -> None:
    """Round-trip check: enqueue with a UUID job_id, then fetch by that id and cancel."""
    from rq.job import Job as RQJob

    rq_job = fake_queue.enqueue(
        lambda x: x, "arg", job_id="11111111-1111-1111-1111-111111111111", job_timeout=60
    )
    assert rq_job.id == "11111111-1111-1111-1111-111111111111"
    assert fake_queue.count == 1

    fetched = RQJob.fetch(rq_job.id, connection=fake_queue.connection)
    fetched.cancel()
    fetched.delete()
    assert fake_queue.count == 0
