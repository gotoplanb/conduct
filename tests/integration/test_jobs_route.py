"""Routes under /jobs — POST submit (sync/async), GET, DELETE."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

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
    db, *, task_type: str, preferred="claude-haiku-4-5", fallback="claude-sonnet-4-5"
) -> RoutingRule:
    rule = RoutingRule(
        task_type=task_type,
        preferred_model=preferred,
        fallback_model=fallback,
        sensitivity="public",  # so the cloud client_headers fixture works
        max_tokens=500,
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
