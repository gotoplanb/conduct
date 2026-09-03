"""Gap coverage for routes/ui.py — login `next` handling, list filters,
relative-age buckets, missing-record and conflict branches on the clients and
connectors forms, secrets-key-missing 503s, and the eval review grouping."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from config.settings import get_settings
from models.job import Job
from models.prompt import PromptVersion
from models.shadow import JobShadow
from models.types import JobStatus

_MISSING = "00000000-0000-0000-0000-000000000000"


@pytest.fixture
def no_secrets_key(monkeypatch: pytest.MonkeyPatch):
    """Blank out CONDUCT_SECRETS_KEY (env overrides the .env file) so the
    encrypt path raises SecretsKeyMissing."""
    import secrets_box

    monkeypatch.setenv("CONDUCT_SECRETS_KEY", "")
    get_settings.cache_clear()
    secrets_box._fernet.cache_clear()
    yield
    secrets_box._fernet.cache_clear()
    get_settings.cache_clear()


@pytest.fixture
def no_grafana(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GRAFANA_BASE_URL", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _seed_job(
    db, *, client_id, task_type="ui-gap", prompt="agent details here",
    status=JobStatus.COMPLETE.value, created_at=None,
) -> Job:
    job = Job(
        client_app_id=client_id,
        task_type=task_type,
        sensitivity="internal",
        prompt=prompt,
        status=status,
        model_used="llama3.3:70b",
        response="ok",
        completed_at=datetime.now(UTC),
    )
    if created_at is not None:
        job.created_at = created_at
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


# --- login `next` ----------------------------------------------------------


async def test_login_with_valid_next_redirects_there(client, admin_token) -> None:
    r = await client.get("/ui/login", params={"next": "/ui/clients"})
    assert r.status_code == 200
    r = await client.post(
        "/ui/login",
        data={"admin_key": admin_token, "next": "/ui/clients"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/ui/clients"


async def test_login_offsite_next_falls_back_to_jobs(client, admin_token) -> None:
    # '//evil.com' is scheme-relative — must not become an open redirect.
    r = await client.post(
        "/ui/login",
        data={"admin_key": admin_token, "next": "//evil.com"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/ui/jobs"


# --- jobs list + detail ----------------------------------------------------


async def test_ui_jobs_status_and_q_filters(admin_client, db_session, seeded_client) -> None:
    marker = f"needle-{uuid4().hex[:8]}"
    hit = await _seed_job(db_session, client_id=seeded_client[0].id, prompt=f"has {marker}")
    miss = await _seed_job(
        db_session, client_id=seeded_client[0].id,
        prompt=f"has {marker}", status=JobStatus.FAILED.value,
    )
    r = await admin_client.get("/ui/jobs", params={"status": "complete", "q": marker})
    assert r.status_code == 200
    assert str(hit.id) in r.text
    assert str(miss.id) not in r.text


async def test_ui_jobs_relative_age_hours_and_days(
    admin_client, db_session, seeded_client
) -> None:
    now = datetime.now(UTC)
    await _seed_job(
        db_session, client_id=seeded_client[0].id,
        prompt="hours old", created_at=now - timedelta(hours=2),
    )
    await _seed_job(
        db_session, client_id=seeded_client[0].id,
        prompt="days old", created_at=now - timedelta(days=3),
    )
    r = await admin_client.get("/ui/jobs")
    assert r.status_code == 200
    assert "2h ago" in r.text
    assert "3d ago" in r.text


async def test_ui_job_detail_unknown_404(admin_client) -> None:
    r = await admin_client.get(f"/ui/jobs/{uuid4()}")
    assert r.status_code == 404


async def test_ui_job_detail_without_grafana_link(
    admin_client, db_session, seeded_client, no_grafana
) -> None:
    job = await _seed_job(db_session, client_id=seeded_client[0].id)
    r = await admin_client.get(f"/ui/jobs/{job.id}")
    assert r.status_code == 200
    assert "explore?orgId" not in r.text


# --- clients forms ---------------------------------------------------------


@pytest.fixture
def commit_integrity_error(monkeypatch: pytest.MonkeyPatch):
    """Make the next AsyncSession.commit raise IntegrityError, then restore.

    The duplicate-name handlers guard a unique constraint that the live schema
    does not actually carry on client_apps.name, so the conflict can only be
    simulated at the session boundary."""
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.ext.asyncio import AsyncSession

    real_commit = AsyncSession.commit
    fired = {"done": False}

    async def _fail_once(self):
        if not fired["done"]:
            fired["done"] = True
            raise IntegrityError("INSERT", {}, Exception("duplicate key"))
        return await real_commit(self)

    monkeypatch.setattr(AsyncSession, "commit", _fail_once)
    return fired


async def test_ui_clients_create_duplicate_name_409(
    admin_client, commit_integrity_error
) -> None:
    r = await admin_client.post("/ui/clients", data={"name": "dupe-created"})
    assert r.status_code == 409
    assert "already exists" in r.text


async def test_ui_clients_toggle_missing_404(admin_client) -> None:
    r = await admin_client.post(f"/ui/clients/{_MISSING}/toggle")
    assert r.status_code == 404
    assert "Client not found" in r.text


async def test_ui_clients_edit_duplicate_name_409(
    admin_client, seeded_client, commit_integrity_error
) -> None:
    r = await admin_client.post(
        f"/ui/clients/{seeded_client[0].id}/edit",
        data={"name": "dupe-renamed"},
    )
    assert r.status_code == 409
    assert "already exists" in r.text


async def test_ui_anthropic_key_missing_client_404(admin_client) -> None:
    r = await admin_client.post(
        f"/ui/clients/{_MISSING}/anthropic-key", data={"api_key": "sk-ant-x"}
    )
    assert r.status_code == 404


async def test_ui_anthropic_key_blank_400(admin_client, seeded_client) -> None:
    r = await admin_client.post(
        f"/ui/clients/{seeded_client[0].id}/anthropic-key", data={"api_key": "   "}
    )
    assert r.status_code == 400
    assert "API key is required" in r.text


async def test_ui_anthropic_key_no_master_key_503(
    admin_client, seeded_client, no_secrets_key
) -> None:
    r = await admin_client.post(
        f"/ui/clients/{seeded_client[0].id}/anthropic-key", data={"api_key": "sk-ant-x"}
    )
    assert r.status_code == 503
    assert "CONDUCT_SECRETS_KEY" in r.text


async def test_ui_clear_anthropic_key_missing_client_404(admin_client) -> None:
    r = await admin_client.post(f"/ui/clients/{_MISSING}/anthropic-key/clear")
    assert r.status_code == 404


async def test_ui_bedrock_creds_missing_client_404(admin_client) -> None:
    r = await admin_client.post(
        f"/ui/clients/{_MISSING}/bedrock-creds",
        data={"bearer_token": "ABSK-x", "region": "us-east-1"},
    )
    assert r.status_code == 404


async def test_ui_bedrock_creds_no_master_key_503(
    admin_client, seeded_client, no_secrets_key
) -> None:
    r = await admin_client.post(
        f"/ui/clients/{seeded_client[0].id}/bedrock-creds",
        data={"bearer_token": "ABSK-x", "region": "us-east-1"},
    )
    assert r.status_code == 503
    assert "CONDUCT_SECRETS_KEY" in r.text


async def test_ui_clear_bedrock_creds_missing_client_404(admin_client) -> None:
    r = await admin_client.post(f"/ui/clients/{_MISSING}/bedrock-creds/clear")
    assert r.status_code == 404


# --- connectors ------------------------------------------------------------


async def test_ui_connectors_create_bad_client_app_id_400(admin_client) -> None:
    r = await admin_client.post(
        "/ui/connectors",
        data={"name": "bad-bind", "client_app_id": "not-a-uuid"},
    )
    assert r.status_code == 400
    assert "Pick a client" in r.text


async def test_ui_connectors_toggle_missing_404(admin_client) -> None:
    r = await admin_client.post(f"/ui/connectors/{_MISSING}/toggle")
    assert r.status_code == 404
    assert "Connector not found" in r.text


# --- tasks history + eval partial ------------------------------------------


async def test_ui_task_history_client_scoped(admin_client, db_session, seeded_client) -> None:
    tt = f"scoped-{uuid4().hex[:8]}"
    db_session.add(
        PromptVersion(
            task_type=tt, client_id=seeded_client[0].id,
            content="v1", edited_by="scoped-editor",
        )
    )
    db_session.add(
        PromptVersion(task_type=tt, client_id=None, content="v1", edited_by="shared-editor")
    )
    await db_session.commit()

    r = await admin_client.get(
        f"/ui/tasks/{tt}/history", params={"client_id": str(seeded_client[0].id)}
    )
    assert r.status_code == 200
    assert "scoped-editor" in r.text
    assert "shared-editor" not in r.text


async def test_ui_eval_partial(admin_client, db_session, seeded_client) -> None:
    await _seed_job(db_session, client_id=seeded_client[0].id, task_type="eval-part")
    r = await admin_client.get("/ui/eval/partial", params={"task_type": "eval-part"})
    assert r.status_code == 200


# --- eval review grouping --------------------------------------------------


async def _seed_shadow(db, *, parent_job_id, model) -> JobShadow:
    s = JobShadow(
        parent_job_id=parent_job_id,
        model=model,
        provider="ollama",
        status=JobStatus.COMPLETE.value,
        response="shadow answer",
        completed_at=datetime.now(UTC),
    )
    db.add(s)
    await db.commit()
    return s


async def test_ui_eval_review_groups_shadows_and_caps_at_ten(
    admin_client, db_session, seeded_client
) -> None:
    """One prompt with two shadows renders as a single group; the page caps at
    10 parent jobs, so the 11th (oldest) is dropped."""
    tt = f"review-{uuid4().hex[:8]}"
    now = datetime.now(UTC)
    jobs = []
    for i in range(11):
        job = await _seed_job(
            db_session, client_id=seeded_client[0].id, task_type=tt,
            prompt=f"prompt {i}", created_at=now - timedelta(minutes=i),
        )
        await _seed_shadow(db_session, parent_job_id=job.id, model=f"m-{i}")
        jobs.append(job)
    # Second shadow on the newest job — exercises the same-group merge.
    await _seed_shadow(db_session, parent_job_id=jobs[0].id, model="m-extra")

    r = await admin_client.get("/ui/eval/review", params={"task_type": tt})
    assert r.status_code == 200
    assert "m-extra" in r.text  # merged into the newest job's group
    assert str(jobs[0].id) in r.text
    assert str(jobs[10].id) not in r.text  # 11th parent dropped by the cap
