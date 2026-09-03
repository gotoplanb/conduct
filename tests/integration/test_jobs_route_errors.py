"""Gap coverage for routes/jobs.py — fan-out validation and inline execution,
the media enqueue path, queue/provider failure translation, list filters, and
the pending-job cancel branches."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from config.settings import get_settings
from models.job import Job
from models.routing import RoutingRule
from models.types import JobStatus
from providers.base import ProviderError


@pytest.fixture
def task_type() -> str:
    return f"test-{uuid4().hex[:8]}"


@pytest.fixture
def resident_models(monkeypatch: pytest.MonkeyPatch):
    """Pin a known resident set so the sync/fan-out paths are deterministic
    regardless of the .env value or cache state left by other tests."""
    monkeypatch.setenv("RESIDENT_MODELS", "llama3.2:3b,gemma4:e4b")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _seed_rule(
    db,
    *,
    task_type: str,
    preferred: str,
    fallback: str | None = None,
    sensitivity: str = "public",
    media_kind: str = "text",
) -> RoutingRule:
    rule = RoutingRule(
        task_type=task_type,
        preferred_model=preferred,
        fallback_model=fallback or preferred,
        sensitivity=sensitivity,
        max_tokens=200,
        media_kind=media_kind,
    )
    db.add(rule)
    await db.commit()
    return rule


async def _seed_job(db, *, client_id, task_type: str, status: str, prompt: str = "hi") -> Job:
    job = Job(
        client_app_id=client_id,
        task_type=task_type,
        sensitivity="public",
        prompt=prompt,
        status=status,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


class _RaisingQueue:
    def enqueue(self, *args, **kwargs):
        raise RuntimeError("redis is down")


# --- fan-out ---------------------------------------------------------------


async def test_fanout_runs_secondaries_inline(
    client, db_session, cloud_client, resident_models, task_type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resident primary with resident + permitted-cloud fan-out targets runs
    sync: the request forces the inline path and the secondaries are dispatched
    with the validated targets. The secondary runner is stubbed — the real one
    shares the request session with the primary under asyncio.gather, which is
    not safe against the test doubles' instant completions."""
    import routes.jobs as jobs_route

    captured: dict = {}

    async def _fake_secondaries(**kwargs):
        captured["models"] = kwargs["secondary_models"]
        captured["parent_id"] = kwargs["parent"].id
        return []

    monkeypatch.setattr(jobs_route, "run_fanout_secondaries", _fake_secondaries)
    await _seed_rule(db_session, task_type=task_type, preferred="llama3.2:3b")
    r = await client.post(
        "/jobs",
        json={
            "task_type": task_type, "prompt": "hi", "system_prompt": "sys",
            "fanout": ["gemma4:e4b", "claude-haiku-4-5"],
        },
        headers={"Authorization": f"Bearer {cloud_client[1]}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "complete"
    assert body["model_used"] == "llama3.2:3b"
    assert captured["models"] == ["gemma4:e4b", "claude-haiku-4-5"]
    assert str(captured["parent_id"]) == body["job_id"]


async def test_fanout_nonresident_primary_400(
    client, db_session, seeded_client, resident_models, task_type
) -> None:
    # Primary routes to a local model outside RESIDENT_MODELS — the API can't
    # call it inline, so fan-out must be refused up front.
    await _seed_rule(db_session, task_type=task_type, preferred="llama3.3:70b")
    r = await client.post(
        "/jobs",
        json={"task_type": task_type, "prompt": "hi", "fanout": ["gemma4:e4b"]},
        headers={"Authorization": f"Bearer {seeded_client[1]}"},
    )
    assert r.status_code == 400
    assert "non-resident local" in r.json()["detail"]


async def test_fanout_cloud_target_confidential_400(
    client, db_session, seeded_client, resident_models, task_type
) -> None:
    await _seed_rule(
        db_session, task_type=task_type, preferred="llama3.2:3b", sensitivity="confidential"
    )
    r = await client.post(
        "/jobs",
        json={"task_type": task_type, "prompt": "s", "fanout": ["claude-haiku-4-5"]},
        headers={"Authorization": f"Bearer {seeded_client[1]}"},
    )
    assert r.status_code == 400
    assert "disallowed for confidential" in r.json()["detail"]


async def test_fanout_cloud_target_internal_without_optin_400(
    client, db_session, seeded_client, resident_models, task_type
) -> None:
    # seeded_client has allow_cloud_for_internal=False.
    await _seed_rule(
        db_session, task_type=task_type, preferred="llama3.2:3b", sensitivity="internal"
    )
    r = await client.post(
        "/jobs",
        json={"task_type": task_type, "prompt": "s", "fanout": ["claude-haiku-4-5"]},
        headers={"Authorization": f"Bearer {seeded_client[1]}"},
    )
    assert r.status_code == 400
    assert "allow_cloud_for_internal" in r.json()["detail"]


# --- sync execution errors -------------------------------------------------


async def test_sync_provider_not_registered_503(
    client, db_session, seeded_client, task_type
) -> None:
    """A Bedrock model is routable (the client has creds) but the API process
    has no Bedrock provider registered — the sync path must 503, not crash."""
    capp, key = seeded_client
    capp.bedrock_creds_encrypted = "opaque-blob"  # only truthiness is consulted here
    await db_session.commit()
    await _seed_rule(db_session, task_type=task_type, preferred="llama3.2:3b")
    r = await client.post(
        "/jobs",
        json={
            "task_type": task_type, "prompt": "hi", "system_prompt": "sys",
            "model": "anthropic.claude-sonnet-4-5",
        },
        headers={"Authorization": f"Bearer {key}"},
    )
    assert r.status_code == 503
    assert "bedrock" in r.json()["detail"]


async def test_sync_provider_error_translates_to_502(
    client, db_session, cloud_client, task_type, monkeypatch: pytest.MonkeyPatch
) -> None:
    import routes.jobs as jobs_route

    await _seed_rule(db_session, task_type=task_type, preferred="claude-haiku-4-5")

    async def _boom(**kwargs):
        raise ProviderError("upstream down")

    monkeypatch.setattr(jobs_route, "execute_job", _boom)
    r = await client.post(
        "/jobs",
        json={"task_type": task_type, "prompt": "hi", "system_prompt": "sys"},
        headers={"Authorization": f"Bearer {cloud_client[1]}"},
    )
    assert r.status_code == 502
    assert "upstream down" in r.json()["detail"]


async def test_sync_failed_primary_returns_failed_jobout(
    client, db_session, cloud_client, task_type, stub_registry
) -> None:
    """When primary and fallback both fail, execute_job records the failure on
    the row without raising — the route returns the failed JobOut (200) and
    must NOT fan out shadows for it."""
    await _seed_rule(
        db_session, task_type=task_type,
        preferred="claude-haiku-4-5", fallback="claude-sonnet-4-5",
    )

    async def _fail(**kwargs):
        raise ProviderError("model down")

    stub_registry.get("anthropic").complete = _fail
    r = await client.post(
        "/jobs",
        json={"task_type": task_type, "prompt": "hi", "system_prompt": "sys"},
        headers={"Authorization": f"Bearer {cloud_client[1]}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failed"
    assert "model down" in body["error"]


# --- enqueue failures ------------------------------------------------------


async def test_async_enqueue_failure_flips_job_to_failed_503(
    client, db_session, seeded_client, fake_redis, task_type, monkeypatch: pytest.MonkeyPatch
) -> None:
    import routes.jobs as jobs_route

    await _seed_rule(db_session, task_type=task_type, preferred="llama3.3:70b")
    monkeypatch.setattr(jobs_route, "get_queue", _RaisingQueue)
    r = await client.post(
        "/jobs",
        json={"task_type": task_type, "prompt": "hi"},
        headers={"Authorization": f"Bearer {seeded_client[1]}"},
    )
    assert r.status_code == 503
    assert "queue backend unavailable" in r.json()["detail"]
    job = await db_session.scalar(select(Job).where(Job.task_type == task_type))
    assert job.status == JobStatus.FAILED.value
    assert "enqueue failed" in job.error


async def test_media_rule_enqueues_on_media_queue(
    client, db_session, seeded_client, fake_redis, task_type, monkeypatch: pytest.MonkeyPatch
) -> None:
    import worker.queue as wq

    # Force the media queue to rebuild against this test's fake redis.
    monkeypatch.setattr(wq, "_media_queue", None)
    await _seed_rule(
        db_session, task_type=task_type, preferred="wander_scene_image", media_kind="image"
    )
    r = await client.post(
        "/jobs",
        json={"task_type": task_type, "prompt": "a quiet map room"},
        headers={"Authorization": f"Bearer {seeded_client[1]}"},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "pending"
    assert body["poll_url"] == f"/jobs/{body['job_id']}"
    job = await db_session.get(Job, UUID(body["job_id"]))
    assert job.status == JobStatus.PENDING.value


async def test_media_enqueue_failure_flips_job_to_failed_503(
    client, db_session, seeded_client, fake_redis, task_type, monkeypatch: pytest.MonkeyPatch
) -> None:
    import worker.queue as wq

    await _seed_rule(
        db_session, task_type=task_type, preferred="wander_scene_image", media_kind="image"
    )
    monkeypatch.setattr(wq, "get_media_queue", _RaisingQueue)
    r = await client.post(
        "/jobs",
        json={"task_type": task_type, "prompt": "a quiet map room"},
        headers={"Authorization": f"Bearer {seeded_client[1]}"},
    )
    assert r.status_code == 503
    job = await db_session.scalar(select(Job).where(Job.task_type == task_type))
    assert job.status == JobStatus.FAILED.value
    assert "enqueue failed" in job.error


# --- admin list filters ----------------------------------------------------


async def test_admin_list_jobs_status_q_and_limit(
    client, db_session, seeded_client, admin_headers, task_type
) -> None:
    marker = f"needle-{uuid4().hex[:8]}"
    for _ in range(2):
        await _seed_job(
            db_session, client_id=seeded_client[0].id, task_type=task_type,
            status=JobStatus.COMPLETE.value, prompt=f"prompt with {marker}",
        )
    await _seed_job(
        db_session, client_id=seeded_client[0].id, task_type=task_type,
        status=JobStatus.FAILED.value, prompt=f"prompt with {marker}",
    )
    r = await client.get(
        f"/jobs?status=complete&q={marker}&limit=1", headers=admin_headers
    )
    assert r.status_code == 200
    jobs = r.json()["jobs"]
    # limit caps the post-filter output even though two rows match.
    assert len(jobs) == 1
    assert jobs[0]["status"] == "complete"


# --- eval + cancel edge cases ----------------------------------------------


async def test_submit_eval_unknown_job_404(client) -> None:
    r = await client.post(
        f"/jobs/{uuid4()}/eval", json={"eval_token": "cdt_ev_x", "score": 3}
    )
    assert r.status_code == 404


async def test_cancel_unknown_job_404(client, seeded_client) -> None:
    r = await client.delete(
        f"/jobs/{uuid4()}", headers={"Authorization": f"Bearer {seeded_client[1]}"}
    )
    assert r.status_code == 404


async def test_cancel_pending_job_with_rq_record(
    client, db_session, seeded_client, fake_redis, task_type
) -> None:
    from rq import Queue
    from rq.job import Job as RQJob

    job = await _seed_job(
        db_session, client_id=seeded_client[0].id, task_type=task_type,
        status=JobStatus.PENDING.value,
    )
    q = Queue("conduct", connection=fake_redis)
    q.enqueue(lambda: None, job_id=str(job.id), job_timeout=60)

    r = await client.delete(
        f"/jobs/{job.id}", headers={"Authorization": f"Bearer {seeded_client[1]}"}
    )
    assert r.status_code == 200
    assert r.json()["status"] == JobStatus.CANCELLED.value
    # The RQ record was cancelled and deleted.
    from rq.exceptions import NoSuchJobError

    with pytest.raises(NoSuchJobError):
        RQJob.fetch(str(job.id), connection=fake_redis)


async def test_cancel_pending_job_without_rq_record(
    client, db_session, seeded_client, fake_redis, task_type
) -> None:
    # No RQ record (worker already pulled it, or TTL expired) — cancel still lands.
    job = await _seed_job(
        db_session, client_id=seeded_client[0].id, task_type=task_type,
        status=JobStatus.PENDING.value,
    )
    r = await client.delete(
        f"/jobs/{job.id}", headers={"Authorization": f"Bearer {seeded_client[1]}"}
    )
    assert r.status_code == 200
    assert r.json()["status"] == JobStatus.CANCELLED.value


async def test_cancel_terminal_job_is_a_noop(
    client, db_session, seeded_client, task_type
) -> None:
    # Neither running nor pending — nothing to cancel; the row is returned as-is.
    job = await _seed_job(
        db_session, client_id=seeded_client[0].id, task_type=task_type,
        status=JobStatus.COMPLETE.value,
    )
    r = await client.delete(
        f"/jobs/{job.id}", headers={"Authorization": f"Bearer {seeded_client[1]}"}
    )
    assert r.status_code == 200
    assert r.json()["status"] == JobStatus.COMPLETE.value


async def test_cancel_pending_job_survives_redis_error(
    client, db_session, seeded_client, fake_redis, task_type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redis hiccups during cancel must not fail the API — the DB row is the
    source of truth and still flips to cancelled."""
    import routes.jobs as jobs_route

    job = await _seed_job(
        db_session, client_id=seeded_client[0].id, task_type=task_type,
        status=JobStatus.PENDING.value,
    )

    def _redis_down():
        raise RuntimeError("redis unreachable")

    monkeypatch.setattr(jobs_route, "get_redis", _redis_down)
    r = await client.delete(
        f"/jobs/{job.id}", headers={"Authorization": f"Bearer {seeded_client[1]}"}
    )
    assert r.status_code == 200
    assert r.json()["status"] == JobStatus.CANCELLED.value


async def test_admin_list_jobs_no_task_type_filter(
    client, db_session, seeded_client, admin_headers, task_type
) -> None:
    """Listing without task_type still returns recent jobs (the unfiltered
    branch of the query builder)."""
    job = await _seed_job(
        db_session, client_id=seeded_client[0].id, task_type=task_type,
        status=JobStatus.COMPLETE.value,
    )
    # created_at in the future of other rows isn't guaranteed; just assert the
    # job is present in a generous window.
    job.created_at = datetime.now(UTC) + timedelta(seconds=1)
    await db_session.commit()
    r = await client.get("/jobs?limit=500", headers=admin_headers)
    assert r.status_code == 200
    assert str(job.id) in [j["job_id"] for j in r.json()["jobs"]]
