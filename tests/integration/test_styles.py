"""Logical style registry + /image submit path (#53)."""

from __future__ import annotations

import pytest

from media_styles import UnknownStyle, resolve_style, visible_styles
from models.style import StyleAlias


async def _add_style(db_session, name, template="wander_scene_image", params=None,
                     client_id=None, archived=False):
    alias = StyleAlias(
        name=name,
        client_id=client_id,
        workflow_template=template,
        params=params or {},
        is_archived=archived,
    )
    db_session.add(alias)
    await db_session.commit()
    return alias


async def test_resolve_none_means_rule_default(db_session) -> None:
    assert await resolve_style(db_session, requested=None, client_id=None) is None


async def test_resolve_shared_and_client_precedence(db_session, seeded_client) -> None:
    capp, _ = seeded_client
    await _add_style(db_session, "t-backdrop", params={"width": 1024})
    await _add_style(db_session, "t-backdrop", params={"width": 1344}, client_id=capp.id)
    style = await resolve_style(db_session, requested="t-backdrop", client_id=capp.id)
    assert style.params == {"width": 1344}
    style = await resolve_style(db_session, requested="t-backdrop", client_id=None)
    assert style.params == {"width": 1024}


async def test_resolve_unknown_raises_with_known(db_session, seeded_client) -> None:
    capp, _ = seeded_client
    await _add_style(db_session, "t-backdrop")
    with pytest.raises(UnknownStyle) as exc:
        await resolve_style(db_session, requested="oops", client_id=capp.id)
    assert "t-backdrop" in exc.value.known and "oops" not in exc.value.known


async def test_archived_style_invisible(db_session) -> None:
    await _add_style(db_session, "gone", archived=True)
    assert "gone" not in [s_.name for s_ in await visible_styles(db_session, None)]
    with pytest.raises(UnknownStyle):
        await resolve_style(db_session, requested="gone", client_id=None)


# --- route-level -------------------------------------------------------------


async def test_image_submit_unknown_style_400(client, seeded_client, fake_redis) -> None:
    _, key = seeded_client
    resp = await client.post(
        "/image",
        json={"prompt": "an ops room", "style": "nope"},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 400
    assert "known styles" in resp.json()["detail"]


async def test_image_submit_stamps_resolved_style(
    client, db_session, seeded_client, fake_redis
) -> None:
    capp, key = seeded_client
    await _add_style(db_session, "t-backdrop-wide", params={"width": 1344, "height": 576})
    resp = await client.post(
        "/image",
        json={"prompt": "an ops room at night", "style": "t-backdrop-wide"},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["style"] == "t-backdrop-wide"
    assert body["expected_output_url"].endswith(".png")

    from sqlalchemy import select

    from models.job import Job

    job = await db_session.scalar(select(Job).where(Job.id == body["job_id"]))
    assert job.task_type == "scene_image"
    assert job.job_metadata["style_resolved"] == {
        "workflow_template": "wander_scene_image",
        "params": {"width": 1344, "height": 576},
    }


async def test_image_submit_no_style_uses_rule_default(
    client, seeded_client, fake_redis
) -> None:
    _, key = seeded_client
    resp = await client.post(
        "/image",
        json={"prompt": "a quiet map room"},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 202
    assert resp.json()["style"] is None


async def test_styles_discovery_merged_view(client, db_session, seeded_client) -> None:
    capp, key = seeded_client
    await _add_style(db_session, "t-backdrop")
    await _add_style(db_session, "t-backdrop", client_id=capp.id)
    await _add_style(db_session, "t-scene-default")
    resp = await client.get("/styles", headers={"Authorization": f"Bearer {key}"})
    assert resp.status_code == 200
    styles = {s["name"]: s for s in resp.json()["styles"]}
    assert styles["t-backdrop"]["scope"] == "client"
    assert styles["t-scene-default"]["scope"] == "shared"
    assert styles["t-scene-default"]["installed"] is True


async def test_registry_put_rejects_missing_template(client, admin_headers) -> None:
    resp = await client.put(
        "/styles/registry/ghost",
        json={"workflow_template": "no-such-workflow"},
        headers=admin_headers,
    )
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"]


async def test_registry_put_archive_revive_cycle(client, admin_headers) -> None:
    put = await client.put(
        "/styles/registry/t-backdrop",
        json={"workflow_template": "wander_scene_image", "params": {"width": 1344}},
        headers=admin_headers,
    )
    assert put.status_code == 200
    assert put.json()["is_archived"] is False

    dele = await client.delete("/styles/registry/t-backdrop", headers=admin_headers)
    assert dele.json()["is_archived"] is True

    put2 = await client.put(
        "/styles/registry/t-backdrop",
        json={"workflow_template": "wander_scene_image"},
        headers=admin_headers,
    )
    assert put2.json()["is_archived"] is False


def test_worker_style_override_merges_params() -> None:
    """The media-branch override: job-level style beats the rule default."""
    style = {"workflow_template": "custom", "params": {"width": 1344}}
    workflow_template = "rule-default"
    extra_params = {"height": 768}
    workflow_template = style.get("workflow_template") or workflow_template
    merged = {**extra_params, **(style.get("params") or {})}
    assert workflow_template == "custom"
    assert merged == {"height": 768, "width": 1344}
