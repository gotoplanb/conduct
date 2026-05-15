"""Routes under /ui — auth, jobs list, job detail, eval."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from models.job import Job


async def _seed_job(db, *, client_id, model="llama3.3:70b", task="bio_generation"):
    job = Job(
        client_app_id=client_id,
        task_type=task,
        sensitivity="internal",
        prompt="agent details here",
        status="complete",
        model_used=model,
        latency_ms=100,
        tokens_in=10,
        tokens_out=20,
        cost_usd=Decimal("0"),
        completed_at=datetime.now(UTC),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def test_ui_root_unauth_redirects_to_login(client) -> None:
    r = await client.get("/ui", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/ui/login"


async def test_ui_root_authed_redirects_to_jobs(client, admin_token) -> None:
    r = await client.get(
        "/ui",
        cookies={"conduct_admin": admin_token},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/ui/jobs"


async def test_ui_jobs_unauth_redirects(client) -> None:
    r = await client.get("/ui/jobs", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/ui/login"


async def test_ui_login_form_renders(client) -> None:
    r = await client.get("/ui/login")
    assert r.status_code == 200
    assert "Conduct" in r.text
    assert "admin_key" in r.text


async def test_ui_login_wrong_key_401(client) -> None:
    r = await client.post(
        "/ui/login", data={"admin_key": "wrong"}, follow_redirects=False
    )
    assert r.status_code == 401


async def test_ui_login_correct_key_sets_cookie_and_redirects(
    client, admin_token
) -> None:
    r = await client.post(
        "/ui/login", data={"admin_key": admin_token}, follow_redirects=False
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/ui/jobs"
    assert "conduct_admin" in r.cookies


async def test_ui_jobs_lists_real_jobs(client, db_session, seeded_client, admin_token) -> None:
    job = await _seed_job(db_session, client_id=seeded_client[0].id)
    r = await client.get(
        "/ui/jobs",
        cookies={"conduct_admin": admin_token},
    )
    assert r.status_code == 200
    assert str(job.id) in r.text
    assert "bio_generation" in r.text


async def test_ui_job_detail(client, db_session, seeded_client, admin_token) -> None:
    job = await _seed_job(db_session, client_id=seeded_client[0].id)
    r = await client.get(
        f"/ui/jobs/{job.id}",
        cookies={"conduct_admin": admin_token},
    )
    assert r.status_code == 200
    assert "bio_generation" in r.text
    assert "llama3.3:70b" in r.text


async def test_ui_jobs_partial_html_fragment(
    client, db_session, seeded_client, admin_token
) -> None:
    await _seed_job(db_session, client_id=seeded_client[0].id, task="alpha")
    r = await client.get(
        "/ui/jobs/partial?task_type=alpha",
        cookies={"conduct_admin": admin_token},
    )
    assert r.status_code == 200
    # Partial returns just the table, not a full HTML page
    assert "<table" in r.text


async def test_ui_eval_page(client, db_session, seeded_client, admin_token) -> None:
    await _seed_job(db_session, client_id=seeded_client[0].id, task="bio_generation")
    r = await client.get(
        "/ui/eval",
        cookies={"conduct_admin": admin_token},
    )
    assert r.status_code == 200


async def test_ui_logout_clears_cookie(client, admin_token) -> None:
    r = await client.post(
        "/ui/logout",
        cookies={"conduct_admin": admin_token},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/ui/login"
