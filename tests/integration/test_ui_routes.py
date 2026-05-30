"""Routes under /ui — auth, jobs list, job detail, eval."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from models.job import Job
from models.prompt import Prompt, PromptVersion
from models.routing import RoutingRule


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


async def test_ui_root_authed_redirects_to_jobs(admin_client) -> None:
    r = await admin_client.get("/ui", follow_redirects=False)
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


async def test_ui_jobs_lists_real_jobs(admin_client, db_session, seeded_client) -> None:
    job = await _seed_job(db_session, client_id=seeded_client[0].id)
    r = await admin_client.get("/ui/jobs")
    assert r.status_code == 200
    assert str(job.id) in r.text
    assert "bio_generation" in r.text


async def test_ui_job_detail(admin_client, db_session, seeded_client) -> None:
    job = await _seed_job(db_session, client_id=seeded_client[0].id)
    r = await admin_client.get(f"/ui/jobs/{job.id}")
    assert r.status_code == 200
    assert "bio_generation" in r.text
    assert "llama3.3:70b" in r.text


async def test_ui_jobs_partial_html_fragment(
    admin_client, db_session, seeded_client
) -> None:
    await _seed_job(db_session, client_id=seeded_client[0].id, task="alpha")
    r = await admin_client.get("/ui/jobs/partial?task_type=alpha")
    assert r.status_code == 200
    # Partial returns just the table, not a full HTML page
    assert "<table" in r.text


async def test_ui_eval_page(admin_client, db_session, seeded_client) -> None:
    await _seed_job(db_session, client_id=seeded_client[0].id, task="bio_generation")
    r = await admin_client.get("/ui/eval")
    assert r.status_code == 200


async def test_ui_clients_unauth_redirects(client) -> None:
    r = await client.get("/ui/clients", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/ui/login"


async def test_ui_clients_lists_clients(admin_client, seeded_client) -> None:
    r = await admin_client.get("/ui/clients")
    assert r.status_code == 200
    assert seeded_client[0].name in r.text


async def test_ui_clients_create_reveals_key_once(admin_client) -> None:
    r = await admin_client.post(
        "/ui/clients",
        data={"name": "ui-created", "notes": "from ui"},
    )
    assert r.status_code == 200
    assert "ui-created" in r.text
    # The raw key is rendered once in the reveal banner.
    assert "cdt_" in r.text
    assert "shown only once" in r.text


async def test_ui_clients_create_blank_name_is_400(admin_client) -> None:
    r = await admin_client.post(
        "/ui/clients",
        data={"name": "   "},
    )
    assert r.status_code == 400
    assert "Name is required" in r.text


async def test_ui_clients_rotate_shows_new_key(admin_client, seeded_client) -> None:
    row, _ = seeded_client
    r = await admin_client.post(f"/ui/clients/{row.id}/rotate")
    assert r.status_code == 200
    assert "rotated" in r.text
    assert "cdt_" in r.text


async def test_ui_clients_toggle_flips_active(admin_client, seeded_client) -> None:
    row, _ = seeded_client
    r = await admin_client.post(f"/ui/clients/{row.id}/toggle")
    assert r.status_code == 200
    assert "now inactive" in r.text


async def test_ui_clients_rotate_missing_is_404(admin_client) -> None:
    r = await admin_client.post("/ui/clients/00000000-0000-0000-0000-000000000000/rotate")
    assert r.status_code == 404


async def test_ui_edit_updates_all_fields(
    admin_client, seeded_client, db_session
) -> None:
    row, _ = seeded_client
    r = await admin_client.post(
        f"/ui/clients/{row.id}/edit",
        data={
            "name": "renamed-client",
            "notes": "edited notes",
            "rate_limit_per_minute": "60",
            "allow_cloud_for_internal": "true",
        },
    )
    assert r.status_code == 200
    assert "Updated renamed-client" in r.text
    await db_session.refresh(row)
    assert row.name == "renamed-client"
    assert row.notes == "edited notes"
    assert row.rate_limit_per_minute == 60
    assert row.allow_cloud_for_internal is True


async def test_ui_edit_clears_rate_limit_when_blank(
    admin_client, seeded_client, db_session
) -> None:
    row, _ = seeded_client
    row.rate_limit_per_minute = 30
    await db_session.commit()
    r = await admin_client.post(
        f"/ui/clients/{row.id}/edit",
        data={"name": row.name, "notes": "", "rate_limit_per_minute": ""},
    )
    assert r.status_code == 200
    await db_session.refresh(row)
    assert row.rate_limit_per_minute is None


async def test_ui_edit_requires_name(
    admin_client, seeded_client
) -> None:
    row, _ = seeded_client
    r = await admin_client.post(
        f"/ui/clients/{row.id}/edit",
        data={"name": "   ", "notes": ""},
    )
    assert r.status_code == 400
    assert "Name is required" in r.text


async def test_ui_edit_missing_is_404(admin_client) -> None:
    r = await admin_client.post(
        "/ui/clients/00000000-0000-0000-0000-000000000000/edit",
        data={"name": "x"},
    )
    assert r.status_code == 404


async def test_ui_set_anthropic_key_flashes_and_persists(
    admin_client, seeded_client, secrets_key, db_session
) -> None:
    row, _ = seeded_client
    r = await admin_client.post(
        f"/ui/clients/{row.id}/anthropic-key",
        data={"api_key": "sk-ant-ui-test"},
    )
    assert r.status_code == 200
    assert f"Anthropic key set for {row.name}" in r.text
    # Plaintext must not appear in the page response
    assert "sk-ant-ui-test" not in r.text
    await db_session.refresh(row)
    assert row.anthropic_api_key_encrypted is not None


async def test_ui_clear_anthropic_key_nulls_columns(
    admin_client, seeded_client, secrets_key, db_session
) -> None:
    row, _ = seeded_client
    await admin_client.post(
        f"/ui/clients/{row.id}/anthropic-key", data={"api_key": "sk-ant-x"}
    )
    r = await admin_client.post(f"/ui/clients/{row.id}/anthropic-key/clear")
    assert r.status_code == 200
    assert f"Anthropic key cleared for {row.name}" in r.text
    await db_session.refresh(row)
    assert row.anthropic_api_key_encrypted is None


async def test_ui_tasks_unauth_redirects(client) -> None:
    r = await client.get("/ui/tasks", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/ui/login"


async def test_ui_tasks_lists_rule_and_shared_prompt(
    admin_client, db_session
) -> None:
    db_session.add(
        RoutingRule(
            task_type="marketing_blurb",
            preferred_model="llama3.3:70b",
            fallback_model="claude-sonnet-4-5",
            sensitivity="internal",
        )
    )
    db_session.add(
        Prompt(task_type="marketing_blurb", client_id=None, content="be punchy")
    )
    await db_session.commit()

    r = await admin_client.get("/ui/tasks")
    assert r.status_code == 200
    assert "marketing_blurb" in r.text
    assert "llama3.3:70b" in r.text
    assert "shared" in r.text


async def test_ui_tasks_shows_client_override(
    admin_client, db_session, seeded_client
) -> None:
    c, _ = seeded_client
    db_session.add(
        Prompt(task_type="marketing_blurb", client_id=c.id, content="client tone")
    )
    await db_session.commit()

    r = await admin_client.get("/ui/tasks")
    assert r.status_code == 200
    assert c.name in r.text


async def test_ui_task_history_partial(admin_client, db_session) -> None:
    db_session.add(
        PromptVersion(
            task_type="marketing_blurb",
            client_id=None,
            content="v1",
            edited_by="seed",
        )
    )
    await db_session.commit()

    r = await admin_client.get("/ui/tasks/marketing_blurb/history")
    assert r.status_code == 200
    assert "seed" in r.text


async def test_ui_task_history_empty(admin_client) -> None:
    r = await admin_client.get("/ui/tasks/does_not_exist/history")
    assert r.status_code == 200
    assert "no history" in r.text


async def test_ui_logout_clears_cookie(admin_client) -> None:
    r = await admin_client.post("/ui/logout", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/ui/login"
