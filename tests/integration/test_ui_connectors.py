"""Routes under /ui/connectors — OAuth-client (MCP connector) management,
plus the resolve_access_token kill-switch when a connector is deactivated."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest_asyncio
from sqlalchemy import select

from models.oauth import OAuthClient, OAuthToken
from oauth_provider import (
    hash_secret,
    new_client_id,
    new_client_secret,
    resolve_access_token,
)

REDIRECT = "https://claude.ai/api/mcp/auth_callback"


@pytest_asyncio.fixture
async def connector(db_session, seeded_client):
    capp, _ = seeded_client
    oc = OAuthClient(
        client_id=new_client_id(),
        client_secret_hash=hash_secret(new_client_secret()),
        name="dave-ios",
        client_app_id=capp.id,
        redirect_uris=[REDIRECT],
    )
    db_session.add(oc)
    await db_session.commit()
    await db_session.refresh(oc)
    return oc


async def test_connectors_unauth_redirects(client) -> None:
    r = await client.get("/ui/connectors", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/ui/login"


async def test_connectors_page_lists(client, connector, admin_token) -> None:
    r = await client.get("/ui/connectors", cookies={"conduct_admin": admin_token})
    assert r.status_code == 200
    assert "dave-ios" in r.text
    assert connector.client_id in r.text
    assert "How to connect in Claude" in r.text


async def test_connectors_create_reveals_pair_once(
    client, db_session, seeded_client, admin_token
) -> None:
    capp, _ = seeded_client
    r = await client.post(
        "/ui/connectors",
        data={"name": "claude-web", "client_app_id": str(capp.id), "redirect_uris": REDIRECT},
        cookies={"conduct_admin": admin_token},
    )
    assert r.status_code == 200
    assert "claude-web" in r.text
    assert "cdtc_" in r.text  # client id
    assert "cdts_" in r.text  # secret, shown once
    assert "shown only once" in r.text

    row = await db_session.scalar(select(OAuthClient).where(OAuthClient.name == "claude-web"))
    assert row is not None
    assert row.redirect_uris == [REDIRECT]


async def test_connectors_create_blank_name_400(client, seeded_client, admin_token) -> None:
    capp, _ = seeded_client
    r = await client.post(
        "/ui/connectors",
        data={"name": "  ", "client_app_id": str(capp.id)},
        cookies={"conduct_admin": admin_token},
    )
    assert r.status_code == 400


async def test_connectors_create_unknown_client_404(client, admin_token) -> None:
    r = await client.post(
        "/ui/connectors",
        data={
            "name": "x",
            "client_app_id": "00000000-0000-0000-0000-000000000000",
        },
        cookies={"conduct_admin": admin_token},
    )
    assert r.status_code == 404


async def test_connectors_create_defaults_redirect(
    client, db_session, seeded_client, admin_token
) -> None:
    capp, _ = seeded_client
    await client.post(
        "/ui/connectors",
        data={"name": "no-redirect", "client_app_id": str(capp.id), "redirect_uris": ""},
        cookies={"conduct_admin": admin_token},
    )
    row = await db_session.scalar(select(OAuthClient).where(OAuthClient.name == "no-redirect"))
    assert row.redirect_uris == [REDIRECT]


async def test_connectors_rotate_secret_changes_hash(
    client, db_session, connector, admin_token
) -> None:
    old_hash = connector.client_secret_hash
    r = await client.post(
        f"/ui/connectors/{connector.id}/rotate-secret",
        cookies={"conduct_admin": admin_token},
    )
    assert r.status_code == 200
    assert "rotated" in r.text
    await db_session.refresh(connector)
    assert connector.client_secret_hash != old_hash


async def test_connectors_toggle(client, db_session, connector, admin_token) -> None:
    r = await client.post(
        f"/ui/connectors/{connector.id}/toggle",
        cookies={"conduct_admin": admin_token},
    )
    assert r.status_code == 200
    assert "now inactive" in r.text


async def test_connectors_rotate_missing_404(client, admin_token) -> None:
    r = await client.post(
        "/ui/connectors/00000000-0000-0000-0000-000000000000/rotate-secret",
        cookies={"conduct_admin": admin_token},
    )
    assert r.status_code == 404


async def _seed_token(db_session, connector) -> str:
    raw = "cdt_at_active_example"
    db_session.add(
        OAuthToken(
            access_token_hash=hash_secret(raw),
            client_id=connector.client_id,
            client_app_id=connector.client_app_id,
            scope="mcp",
            access_expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
    )
    await db_session.commit()
    return raw


async def test_deactivating_connector_revokes_its_tokens(db_session, connector) -> None:
    raw = await _seed_token(db_session, connector)
    # Active connector: token resolves.
    assert await resolve_access_token(db_session, raw) is not None

    connector.is_active = False
    await db_session.commit()
    # Deactivated connector: same token no longer resolves.
    assert await resolve_access_token(db_session, raw) is None
