"""Routes under /jobs — POST submit (sync/async), GET, DELETE."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from models.job import Job
from models.routing import RoutingRule
from models.types import JobStatus


async def _seed_job(db, *, client_id, task_type, status=JobStatus.COMPLETE.value):
    job = Job(
        client_app_id=client_id,
        task_type=task_type,
        sensitivity="public",
        prompt="hi",
        status=status,
        model_used="llama3.2:3b",
        response="ok",
        completed_at=datetime.now(UTC),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


@pytest.fixture
def task_type() -> str:
    """Unique task_type per test so we don't collide with pre-existing dev-DB rules."""
    return f"test-{uuid4().hex[:8]}"


async def _seed_rule(
    db,
    *,
    task_type: str,
    preferred="claude-haiku-4-5",
    fallback="claude-sonnet-4-5",
    eval_shadow_models: list[dict] | None = None,
    sampling: str | None = None,
) -> RoutingRule:
    rule = RoutingRule(
        task_type=task_type,
        preferred_model=preferred,
        fallback_model=fallback,
        sensitivity="public",  # so the cloud client_headers fixture works
        max_tokens=500,
        eval_shadow_models=eval_shadow_models or [],
        sampling=sampling,
    )
    db.add(rule)
    await db.commit()
    return rule


async def test_submit_sync_cloud_job(client, db_session, cloud_client, task_type) -> None:
    await _seed_rule(db_session, task_type=task_type)
    headers = {"Authorization": f"Bearer {cloud_client[1]}"}
    r = await client.post(
        "/jobs",
        # Provide system_prompt so the resolver isn't asked for a prompts/shared/
        # file that doesn't exist for synthetic test task_types.
        json={"task_type": task_type, "prompt": "hello", "system_prompt": "test sys"},
        headers=headers,
    )
    # Cloud goes sync via stub provider; stub returns ProviderResponse
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "complete"
    assert body["model_used"] == "claude-haiku-4-5"


def _capture_complete(stub_registry, provider_name="anthropic"):
    """Wrap a stub provider's complete() to record the kwargs it's called
    with. Returns the dict that gets populated on the next call."""
    captured: dict = {}
    prov = stub_registry.get(provider_name)
    original = prov.complete

    async def _wrapped(**kwargs):
        captured.update(kwargs)
        return await original(**kwargs)

    prov.complete = _wrapped
    return captured


async def test_deterministic_rule_sends_temp0_and_seed_to_provider(
    client, db_session, cloud_client, task_type, stub_registry
) -> None:
    """A deterministic task reaches the provider with temperature 0 and a
    derived (non-null) seed — the reproducibility path."""
    await _seed_rule(db_session, task_type=task_type, sampling="deterministic")
    captured = _capture_complete(stub_registry)
    r = await client.post(
        "/jobs",
        json={"task_type": task_type, "prompt": "a bio", "system_prompt": "context"},
        headers={"Authorization": f"Bearer {cloud_client[1]}"},
    )
    assert r.status_code == 200
    assert captured["temperature"] == 0.0
    assert isinstance(captured["seed"], int)


async def test_creative_rule_sends_high_temp_and_no_seed(
    client, db_session, cloud_client, task_type, stub_registry
) -> None:
    """A creative task reaches the provider with high temperature and no seed
    (so each call varies)."""
    await _seed_rule(db_session, task_type=task_type, sampling="creative")
    captured = _capture_complete(stub_registry)
    r = await client.post(
        "/jobs",
        json={"task_type": task_type, "prompt": "joke about robots", "system_prompt": "s"},
        headers={"Authorization": f"Bearer {cloud_client[1]}"},
    )
    assert r.status_code == 200
    assert captured["temperature"] == 1.0
    assert captured["seed"] is None


async def test_per_request_sampling_overrides_rule_at_provider(
    client, db_session, cloud_client, task_type, stub_registry
) -> None:
    """Rule is deterministic, but the request asks for creative → the provider
    gets the creative params (override wins end-to-end)."""
    await _seed_rule(db_session, task_type=task_type, sampling="deterministic")
    captured = _capture_complete(stub_registry)
    r = await client.post(
        "/jobs",
        json={
            "task_type": task_type, "prompt": "x", "system_prompt": "s",
            "sampling": "creative",
        },
        headers={"Authorization": f"Bearer {cloud_client[1]}"},
    )
    assert r.status_code == 200
    assert captured["temperature"] == 1.0
    assert captured["seed"] is None


async def test_async_job_persists_sampling_override_on_row(
    client, db_session, seeded_client, fake_redis, task_type
) -> None:
    """The async path must persist the per-request sampling on the Job row so
    the worker (which re-decides) can honor it."""
    await _seed_rule(
        db_session, task_type=task_type, preferred="llama3.3:70b", fallback="llama3.3:70b"
    )
    r = await client.post(
        "/jobs",
        json={"task_type": task_type, "prompt": "hi", "sampling": "deterministic"},
        headers={"Authorization": f"Bearer {seeded_client[1]}"},
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    row = await db_session.get(Job, UUID(job_id))
    assert row.sampling == "deterministic"


async def test_submit_async_when_explicitly_requested(
    client, db_session, cloud_client, fake_redis, task_type
) -> None:
    await _seed_rule(db_session, task_type=task_type)
    headers = {"Authorization": f"Bearer {cloud_client[1]}"}
    r = await client.post(
        "/jobs",
        json={"task_type": task_type, "prompt": "hi", "async": True},
        headers=headers,
    )
    assert r.status_code == 202
    assert "job_id" in r.json()
    assert r.json()["status"] == "pending"


async def test_submit_local_model_goes_async(
    client, db_session, seeded_client, fake_redis, task_type
) -> None:
    # local model → worker queue → 202
    await _seed_rule(
        db_session, task_type=task_type, preferred="llama3.3:70b", fallback="llama3.3:70b"
    )
    headers = {"Authorization": f"Bearer {seeded_client[1]}"}
    r = await client.post(
        "/jobs",
        json={"task_type": task_type, "prompt": "hi"},
        headers=headers,
    )
    assert r.status_code == 202


async def test_submit_sensitivity_violation(
    client, db_session, seeded_client, task_type
) -> None:
    # Confidential rule + explicit cloud model override → 400
    rule = RoutingRule(
        task_type=task_type,
        preferred_model="llama3.3:70b",
        fallback_model="llama3.3:70b",
        sensitivity="confidential",
        max_tokens=500,
    )
    db_session.add(rule)
    await db_session.commit()

    headers = {"Authorization": f"Bearer {seeded_client[1]}"}
    r = await client.post(
        "/jobs",
        json={
            "task_type": task_type,
            "prompt": "secret",
            "model": "claude-haiku-4-5",  # cloud override forbidden under confidential
        },
        headers=headers,
    )
    assert r.status_code == 400


async def test_submit_fanout_invalid_target_400(
    client, db_session, cloud_client, task_type
) -> None:
    await _seed_rule(db_session, task_type=task_type)
    headers = {"Authorization": f"Bearer {cloud_client[1]}"}
    r = await client.post(
        "/jobs",
        json={
            "task_type": task_type,
            "prompt": "hi",
            "fanout": ["llama3.3:70b"],  # non-resident local — not allowed
        },
        headers=headers,
    )
    assert r.status_code == 400


async def test_get_job_owner_only(
    client, db_session, seeded_client, cloud_client, task_type
) -> None:
    await _seed_rule(db_session, task_type=task_type, preferred="claude-haiku-4-5")
    # cloud_client submits a job
    r1 = await client.post(
        "/jobs",
        json={"task_type": task_type, "prompt": "hi", "system_prompt": "sys"},
        headers={"Authorization": f"Bearer {cloud_client[1]}"},
    )
    job_id = r1.json()["job_id"]

    # Owner can read
    r2 = await client.get(
        f"/jobs/{job_id}",
        headers={"Authorization": f"Bearer {cloud_client[1]}"},
    )
    assert r2.status_code == 200

    # Non-owner gets 404 (we don't reveal existence)
    r3 = await client.get(
        f"/jobs/{job_id}",
        headers={"Authorization": f"Bearer {seeded_client[1]}"},
    )
    assert r3.status_code == 404


async def test_get_unknown_job_404(client, cloud_client) -> None:
    r = await client.get(
        f"/jobs/{uuid4()}", headers={"Authorization": f"Bearer {cloud_client[1]}"}
    )
    assert r.status_code == 404


async def test_jobs_requires_auth(client) -> None:
    r = await client.post("/jobs", json={"task_type": "x", "prompt": "y"})
    assert r.status_code == 403


async def test_cancel_live_running_job_is_409(
    client, db_session, cloud_client, task_type, fake_redis
) -> None:
    """A running job whose RQ record still exists belongs to a live worker —
    the API must refuse to cancel it, else we race the executor."""
    from rq import Queue

    job = await _seed_job(
        db_session,
        client_id=cloud_client[0].id,
        task_type=task_type,
        status=JobStatus.RUNNING.value,
    )
    # Plant an RQ record with matching id so the route's fetch succeeds.
    q = Queue("conduct", connection=fake_redis)
    q.enqueue(lambda: None, job_id=str(job.id), job_timeout=60)

    r = await client.delete(
        f"/jobs/{job.id}",
        headers={"Authorization": f"Bearer {cloud_client[1]}"},
    )
    assert r.status_code == 409
    assert "running" in r.json()["detail"]


async def test_cancel_abandoned_running_job_flips_to_cancelled(
    client, db_session, cloud_client, task_type, fake_redis
) -> None:
    """When the worker host crashes, the Job row stays status=running but the
    RQ record is gone (AbandonedJobError / TTL). The route should detect the
    NoSuchJobError and flip the orphan to cancelled so it doesn't stick."""
    job = await _seed_job(
        db_session,
        client_id=cloud_client[0].id,
        task_type=task_type,
        status=JobStatus.RUNNING.value,
    )
    # fake_redis is empty — no RQ record for this id → NoSuchJobError path.

    r = await client.delete(
        f"/jobs/{job.id}",
        headers={"Authorization": f"Bearer {cloud_client[1]}"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == JobStatus.CANCELLED.value

    await db_session.refresh(job)
    assert job.status == JobStatus.CANCELLED.value


async def test_cancel_running_job_in_failed_registry_flips_to_cancelled(
    client, db_session, cloud_client, task_type, fake_redis
) -> None:
    """Crash variant where the RQ record survives: on worker-host crash,
    AbandonedJobError moves the job to the FailedJobRegistry, so RQJob.fetch
    still succeeds (no NoSuchJobError) while the Postgres row is stuck at
    running. The route must read the RQ status, see it's failed, and reconcile
    the orphan to cancelled instead of 409'ing forever."""
    from rq import Queue
    from rq.job import JobStatus as RQJobStatus

    job = await _seed_job(
        db_session,
        client_id=cloud_client[0].id,
        task_type=task_type,
        status=JobStatus.RUNNING.value,
    )
    # Plant a fetchable RQ record, then mark it failed (the AbandonedJobError
    # end-state) — the record is NOT gone, so this exercises the is_failed path.
    q = Queue("conduct", connection=fake_redis)
    rq_job = q.enqueue(lambda: None, job_id=str(job.id), job_timeout=60)
    rq_job.set_status(RQJobStatus.FAILED)

    r = await client.delete(
        f"/jobs/{job.id}",
        headers={"Authorization": f"Bearer {cloud_client[1]}"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == JobStatus.CANCELLED.value

    await db_session.refresh(job)
    assert job.status == JobStatus.CANCELLED.value


async def test_admin_list_jobs(client, db_session, seeded_client, admin_headers, task_type) -> None:
    await _seed_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    await _seed_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    r = await client.get(f"/jobs?task_type={task_type}", headers=admin_headers)
    assert r.status_code == 200
    jobs = r.json()["jobs"]
    assert len(jobs) == 2
    assert all(j["task_type"] == task_type for j in jobs)
    assert jobs[0]["client_app"] == seeded_client[0].name


async def test_admin_list_jobs_requires_admin(client, seeded_client) -> None:
    r = await client.get("/jobs", headers={"Authorization": f"Bearer {seeded_client[1]}"})
    assert r.status_code == 403


async def test_admin_can_read_any_job(
    client, db_session, seeded_client, admin_headers, task_type
) -> None:
    job = await _seed_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    r = await client.get(f"/jobs/{job.id}", headers=admin_headers)
    assert r.status_code == 200
    assert r.json()["job_id"] == str(job.id)
    assert r.json()["response"] == "ok"


async def test_admin_list_jobs_score_filter(
    client, db_session, seeded_client, admin_headers, task_type
) -> None:
    cid = seeded_client[0].id
    low = await _seed_job(db_session, client_id=cid, task_type=task_type)
    high = await _seed_job(db_session, client_id=cid, task_type=task_type)
    await _seed_job(db_session, client_id=cid, task_type=task_type)  # unscored
    low.job_metadata = {"quality_scores": [{"score": 2}]}
    high.job_metadata = {"quality_scores": [{"score": 5}]}
    await db_session.commit()

    # max_score=3 returns only the scored-low job (unscored excluded).
    r = await client.get(f"/jobs?task_type={task_type}&max_score=3", headers=admin_headers)
    assert r.status_code == 200
    jobs = r.json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == str(low.id)
    assert jobs[0]["avg_score"] == 2.0
    assert jobs[0]["score_count"] == 1


async def _mint_eval_link(client, job_id, headers) -> str:
    r = await client.post(f"/jobs/{job_id}/eval-link", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["eval_token"].startswith("cdt_ev_")
    return body["eval_token"]


async def test_eval_link_and_token_submit(
    client, db_session, seeded_client, task_type
) -> None:
    job = await _seed_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    headers = {"Authorization": f"Bearer {seeded_client[1]}"}
    token = await _mint_eval_link(client, job.id, headers)

    # Submit a score with the token — no bearer auth.
    r = await client.post(
        f"/jobs/{job.id}/eval", json={"eval_token": token, "score": 4, "note": "good"}
    )
    assert r.status_code == 200
    assert r.json()["recorded"] is True

    # The score landed in quality_scores, tagged via=url.
    await db_session.refresh(job)
    scores = job.job_metadata["quality_scores"]
    assert scores[-1]["score"] == 4
    assert scores[-1]["via"] == "url"


async def test_eval_token_single_use(client, db_session, seeded_client, task_type) -> None:
    job = await _seed_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    headers = {"Authorization": f"Bearer {seeded_client[1]}"}
    token = await _mint_eval_link(client, job.id, headers)
    body = {"eval_token": token, "score": 5}
    assert (await client.post(f"/jobs/{job.id}/eval", json=body)).status_code == 200
    second = await client.post(f"/jobs/{job.id}/eval", json=body)
    assert second.status_code == 409


async def test_eval_invalid_token_401(client, db_session, seeded_client, task_type) -> None:
    job = await _seed_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    headers = {"Authorization": f"Bearer {seeded_client[1]}"}
    await _mint_eval_link(client, job.id, headers)
    r = await client.post(
        f"/jobs/{job.id}/eval", json={"eval_token": "cdt_ev_wrong", "score": 3}
    )
    assert r.status_code == 401


async def test_eval_score_out_of_range_422(client, db_session, seeded_client, task_type) -> None:
    job = await _seed_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    headers = {"Authorization": f"Bearer {seeded_client[1]}"}
    token = await _mint_eval_link(client, job.id, headers)
    r = await client.post(f"/jobs/{job.id}/eval", json={"eval_token": token, "score": 9})
    assert r.status_code == 422


async def test_eval_link_owner_only(
    client, db_session, seeded_client, cloud_client, task_type
) -> None:
    # Job belongs to seeded_client; cloud_client must not be able to mint a link.
    job = await _seed_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    r = await client.post(
        f"/jobs/{job.id}/eval-link",
        headers={"Authorization": f"Bearer {cloud_client[1]}"},
    )
    assert r.status_code == 404


async def _seed_shadow(
    db, *, parent_job_id, model, provider,
    response="r", status=JobStatus.COMPLETE.value,
):
    from datetime import UTC, datetime
    from decimal import Decimal

    from models.shadow import JobShadow

    s = JobShadow(
        parent_job_id=parent_job_id,
        model=model,
        provider=provider,
        status=status,
        response=response,
        tokens_in=10,
        tokens_out=5,
        cost_usd=Decimal("0.001"),
        latency_ms=100,
        completed_at=datetime.now(UTC),
    )
    db.add(s)
    await db.commit()
    await db.refresh(s)
    return s


async def test_list_shadows_returns_all_for_owner(
    client, db_session, seeded_client, task_type
) -> None:
    job = await _seed_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    await _seed_shadow(db_session, parent_job_id=job.id, model="gemma4:e4b", provider="ollama")
    await _seed_shadow(
        db_session, parent_job_id=job.id, model="claude-haiku-4-5", provider="anthropic"
    )
    r = await client.get(
        f"/jobs/{job.id}/shadows",
        headers={"Authorization": f"Bearer {seeded_client[1]}"},
    )
    assert r.status_code == 200
    out = r.json()
    assert out["parent_job_id"] == str(job.id)
    models = sorted(s["model"] for s in out["shadows"])
    assert models == ["claude-haiku-4-5", "gemma4:e4b"]


async def test_list_shadows_non_owner_is_404(
    client, db_session, seeded_client, cloud_client, task_type
) -> None:
    job = await _seed_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    await _seed_shadow(db_session, parent_job_id=job.id, model="x", provider="ollama")
    r = await client.get(
        f"/jobs/{job.id}/shadows",
        headers={"Authorization": f"Bearer {cloud_client[1]}"},
    )
    assert r.status_code == 404


async def test_list_shadows_admin_sees_any_job(
    client, admin_headers, db_session, seeded_client, task_type
) -> None:
    job = await _seed_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    await _seed_shadow(db_session, parent_job_id=job.id, model="x", provider="ollama")
    r = await client.get(f"/jobs/{job.id}/shadows", headers=admin_headers)
    assert r.status_code == 200
    assert len(r.json()["shadows"]) == 1


async def test_list_shadows_unknown_job_is_404(client, admin_headers) -> None:
    r = await client.get(f"/jobs/{uuid4()}/shadows", headers=admin_headers)
    assert r.status_code == 404


async def test_sync_post_enqueues_rule_shadows(
    client, db_session, cloud_client, task_type, fake_redis, monkeypatch
) -> None:
    """Regression: sync POST /jobs must call enqueue_shadows_for_parent
    after the primary lands. Earlier, only the MCP and async-worker paths
    fanned shadows out — the sync HTTP path silently skipped them."""
    import eval.shadow_runner as shadow_runner

    await _seed_rule(
        db_session,
        task_type=task_type,
        eval_shadow_models=[
            {"model": "qwen3.5:9b", "rate": 1.0},
            {"model": "llama3.1:8b", "rate": 1.0},
        ],
    )

    captured: list[dict] = []

    async def _stub_enqueue(*, parent_job, rule, client, session):
        captured.append({
            "parent_job_id": str(parent_job.id),
            "rule_task": rule.task_type,
            "client_id": str(client.id),
        })

    monkeypatch.setattr(shadow_runner, "enqueue_shadows_for_parent", _stub_enqueue)
    import routes.jobs as jobs_route
    monkeypatch.setattr(jobs_route, "enqueue_shadows_for_parent", _stub_enqueue)

    headers = {"Authorization": f"Bearer {cloud_client[1]}"}
    r = await client.post(
        "/jobs",
        json={"task_type": task_type, "prompt": "hi", "system_prompt": "sys"},
        headers=headers,
    )
    assert r.status_code == 200
    assert r.json()["status"] == "complete"
    assert len(captured) == 1
    assert captured[0]["rule_task"] == task_type
    assert captured[0]["parent_job_id"] == r.json()["job_id"]


async def test_sync_post_skips_shadows_when_primary_fails(
    client, db_session, cloud_client, task_type, monkeypatch
) -> None:
    """If the primary fails, we shouldn't fan shadows out against an empty
    job — they'd just compare to nothing. Mirrors the MCP path's behavior."""
    import eval.shadow_runner as shadow_runner

    await _seed_rule(
        db_session,
        task_type=task_type,
        eval_shadow_models=[{"model": "x", "rate": 1.0}],
    )

    called = []

    async def _stub_enqueue(**kwargs):
        called.append(kwargs)

    monkeypatch.setattr(shadow_runner, "enqueue_shadows_for_parent", _stub_enqueue)
    import routes.jobs as jobs_route
    monkeypatch.setattr(jobs_route, "enqueue_shadows_for_parent", _stub_enqueue)

    # Make the primary raise PromptNotFoundError by giving no system_prompt
    # for a task_type that has no Dave-scoped prompt seeded.
    headers = {"Authorization": f"Bearer {cloud_client[1]}"}
    r = await client.post(
        "/jobs",
        json={"task_type": task_type, "prompt": "hi"},
        headers=headers,
    )
    assert r.status_code == 500
    assert called == []
