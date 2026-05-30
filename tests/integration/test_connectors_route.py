"""Admin JSON API for connectors (/connectors). Parallels the /ui/connectors
UI flow and backs `conduct connectors`."""

from __future__ import annotations

from uuid import UUID, uuid4

from models.oauth import OAuthClient


async def test_create_connector_reveals_secret_once(
    client, admin_headers, seeded_client
) -> None:
    capp, _ = seeded_client
    r = await client.post(
        "/connectors",
        json={"name": "test-conn", "client_app_id": str(capp.id)},
        headers=admin_headers,
    )
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "test-conn"
    assert body["client_id"].startswith("cdtc_")
    assert body["client_secret"].startswith("cdts_")
    UUID(body["id"])
    # Default redirect is Claude's callback when none specified.
    assert body["redirect_uris"] == ["https://claude.ai/api/mcp/auth_callback"]


async def test_create_connector_unknown_client_404(client, admin_headers) -> None:
    r = await client.post(
        "/connectors",
        json={"name": "x", "client_app_id": str(uuid4())},
        headers=admin_headers,
    )
    assert r.status_code == 404


async def test_create_connector_requires_admin(client, seeded_client) -> None:
    capp, raw = seeded_client
    r = await client.post(
        "/connectors",
        json={"name": "x", "client_app_id": str(capp.id)},
        headers={"Authorization": f"Bearer {raw}"},
    )
    assert r.status_code == 403


async def test_list_connectors(client, admin_headers, db_session, seeded_client) -> None:
    capp, _ = seeded_client
    db_session.add(
        OAuthClient(
            client_id="cdtc_list_test",
            client_secret_hash="x",
            name="listme",
            client_app_id=capp.id,
            redirect_uris=[],
        )
    )
    await db_session.commit()
    r = await client.get("/connectors", headers=admin_headers)
    assert r.status_code == 200
    assert any(c["name"] == "listme" for c in r.json())


async def test_rotate_secret_changes_hash(
    client, admin_headers, db_session, seeded_client
) -> None:
    capp, _ = seeded_client
    row = OAuthClient(
        client_id="cdtc_rotate_test",
        client_secret_hash="original",
        name="rotateme",
        client_app_id=capp.id,
        redirect_uris=[],
    )
    db_session.add(row)
    await db_session.commit()
    await db_session.refresh(row)

    r = await client.post(f"/connectors/{row.id}/rotate-secret", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["client_secret"].startswith("cdts_")
    await db_session.refresh(row)
    assert row.client_secret_hash != "original"


async def test_rotate_missing_404(client, admin_headers) -> None:
    r = await client.post(f"/connectors/{uuid4()}/rotate-secret", headers=admin_headers)
    assert r.status_code == 404


async def test_patch_connector_toggles_active(
    client, admin_headers, db_session, seeded_client
) -> None:
    capp, _ = seeded_client
    row = OAuthClient(
        client_id="cdtc_toggle_test",
        client_secret_hash="x",
        name="toggleme",
        client_app_id=capp.id,
        redirect_uris=[],
    )
    db_session.add(row)
    await db_session.commit()
    await db_session.refresh(row)

    r = await client.patch(
        f"/connectors/{row.id}", json={"is_active": False}, headers=admin_headers
    )
    assert r.status_code == 200
    assert r.json()["is_active"] is False
