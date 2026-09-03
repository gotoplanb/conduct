"""Routes under /clients — admin CRUD + per-client usage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from models.client import ClientAppUsage


async def test_create_client_returns_raw_key_once(client, admin_headers) -> None:
    r = await client.post(
        "/clients",
        json={"name": "alpha", "notes": "test", "allow_cloud_for_internal": True},
        headers=admin_headers,
    )
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "alpha"
    assert body["api_key"].startswith("cdt_")
    assert body["is_active"] is True
    UUID(body["id"])  # well-formed UUID


async def test_create_client_unauthenticated_is_403(client) -> None:
    r = await client.post("/clients", json={"name": "alpha"})
    assert r.status_code == 403


async def test_create_client_wrong_admin_is_403(client) -> None:
    r = await client.post(
        "/clients", json={"name": "alpha"}, headers={"Authorization": "Bearer wrong"}
    )
    assert r.status_code == 403


async def test_duplicate_name_is_409(client, admin_headers) -> None:
    """client_apps.name carries uq_client_apps_name; the second create trips
    the constraint for real and the handler maps it to 409."""
    first = await client.post("/clients", json={"name": "dup"}, headers=admin_headers)
    assert first.status_code == 201
    r = await client.post("/clients", json={"name": "dup"}, headers=admin_headers)
    assert r.status_code == 409
    assert "conflict" in r.json()["detail"]


async def test_list_clients(client, admin_headers, seeded_client) -> None:
    r = await client.get("/clients", headers=admin_headers)
    assert r.status_code == 200
    names = [c["name"] for c in r.json()]
    assert seeded_client[0].name in names


async def test_patch_client(client, admin_headers, seeded_client) -> None:
    row, _ = seeded_client
    r = await client.patch(
        f"/clients/{row.id}",
        json={"is_active": False, "notes": "patched"},
        headers=admin_headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["is_active"] is False
    assert body["notes"] == "patched"


async def test_patch_nonexistent_client_is_404(client, admin_headers) -> None:
    r = await client.patch(
        "/clients/00000000-0000-0000-0000-000000000000",
        json={"is_active": False},
        headers=admin_headers,
    )
    assert r.status_code == 404


async def test_rotate_key_returns_new_key_and_invalidates_old(
    client, admin_headers, seeded_client
) -> None:
    row, old_key = seeded_client
    r = await client.post(f"/clients/{row.id}/rotate-key", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    new_key = body["api_key"]
    assert new_key.startswith("cdt_")
    assert new_key != old_key
    assert str(row.id) == body["id"]


async def test_rotate_key_nonexistent_client_is_404(client, admin_headers) -> None:
    r = await client.post(
        "/clients/00000000-0000-0000-0000-000000000000/rotate-key",
        headers=admin_headers,
    )
    assert r.status_code == 404


async def test_rotate_key_unauthenticated_is_403(client, seeded_client) -> None:
    row, _ = seeded_client
    r = await client.post(f"/clients/{row.id}/rotate-key")
    assert r.status_code == 403


async def test_list_clients_includes_key_created_at(
    client, admin_headers, seeded_client
) -> None:
    r = await client.get("/clients", headers=admin_headers)
    assert r.status_code == 200
    me = next(c for c in r.json() if c["id"] == str(seeded_client[0].id))
    assert "key_created_at" in me


async def test_usage_aggregates_by_day(
    client, admin_headers, seeded_client, db_session: AsyncSession
) -> None:
    row, _ = seeded_client
    today = datetime.now(UTC).date()
    db_session.add_all(
        [
            ClientAppUsage(
                client_app_id=row.id,
                date=today,
                tokens_in=100,
                tokens_out=50,
                job_count=2,
                cost_usd=Decimal("0.0042"),
            ),
            ClientAppUsage(
                client_app_id=row.id,
                date=today - timedelta(days=1),
                tokens_in=20,
                tokens_out=10,
                job_count=1,
                cost_usd=Decimal("0"),
            ),
        ]
    )
    await db_session.commit()

    r = await client.get(f"/clients/{row.id}/usage?days=7", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["job_count"] == 3
    assert body["tokens_in"] == 120
    assert body["tokens_out"] == 60
    assert len(body["by_day"]) == 2


async def test_usage_nonexistent_client_is_404(client, admin_headers) -> None:
    r = await client.get(
        "/clients/00000000-0000-0000-0000-000000000000/usage", headers=admin_headers
    )
    assert r.status_code == 404


async def test_set_anthropic_key_persists_encrypted(
    client, admin_headers, seeded_client, secrets_key, db_session: AsyncSession
) -> None:
    row, _ = seeded_client
    plaintext = "sk-ant-test-abcdef"
    r = await client.put(
        f"/clients/{row.id}/anthropic-key",
        json={"api_key": plaintext},
        headers=admin_headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["anthropic_api_key_set_at"] is not None
    # Plaintext is never echoed back
    assert "api_key" not in body
    assert plaintext not in r.text

    from secrets_box import decrypt

    await db_session.refresh(row)
    assert row.anthropic_api_key_encrypted is not None
    assert row.anthropic_api_key_encrypted != plaintext
    assert decrypt(row.anthropic_api_key_encrypted) == plaintext


async def test_clear_anthropic_key_nulls_columns(
    client, admin_headers, seeded_client, secrets_key, db_session: AsyncSession
) -> None:
    row, _ = seeded_client
    await client.put(
        f"/clients/{row.id}/anthropic-key",
        json={"api_key": "sk-ant-x"},
        headers=admin_headers,
    )
    r = await client.delete(
        f"/clients/{row.id}/anthropic-key", headers=admin_headers
    )
    assert r.status_code == 200
    assert r.json()["anthropic_api_key_set_at"] is None
    await db_session.refresh(row)
    assert row.anthropic_api_key_encrypted is None
    assert row.anthropic_api_key_set_at is None


async def test_set_anthropic_key_nonexistent_is_404(
    client, admin_headers, secrets_key
) -> None:
    r = await client.put(
        "/clients/00000000-0000-0000-0000-000000000000/anthropic-key",
        json={"api_key": "sk-ant-x"},
        headers=admin_headers,
    )
    assert r.status_code == 404


async def test_set_anthropic_key_unauthenticated_is_403(
    client, seeded_client
) -> None:
    row, _ = seeded_client
    r = await client.put(
        f"/clients/{row.id}/anthropic-key", json={"api_key": "sk-ant-x"}
    )
    assert r.status_code == 403


async def test_set_anthropic_key_no_master_key_is_503(
    client, admin_headers, seeded_client, monkeypatch
) -> None:
    """If CONDUCT_SECRETS_KEY isn't configured, the route should refuse to
    pretend to store the key — operator needs to set the env var first."""
    import secrets_box

    # Empty string overrides .env (pydantic-settings reads .env directly,
    # so plain `delenv` wouldn't simulate "unset" on a dev box that has the
    # key configured).
    monkeypatch.setenv("CONDUCT_SECRETS_KEY", "")
    get_settings.cache_clear()
    secrets_box._fernet.cache_clear()
    row, _ = seeded_client
    r = await client.put(
        f"/clients/{row.id}/anthropic-key",
        json={"api_key": "sk-ant-x"},
        headers=admin_headers,
    )
    assert r.status_code == 503


async def test_list_clients_includes_anthropic_key_set_at(
    client, admin_headers, seeded_client, secrets_key
) -> None:
    row, _ = seeded_client
    await client.put(
        f"/clients/{row.id}/anthropic-key",
        json={"api_key": "sk-ant-x"},
        headers=admin_headers,
    )
    r = await client.get("/clients", headers=admin_headers)
    me = next(c for c in r.json() if c["id"] == str(row.id))
    assert me["anthropic_api_key_set_at"] is not None


async def test_set_bedrock_creds_persists_encrypted(
    client, admin_headers, seeded_client, secrets_key, db_session: AsyncSession
) -> None:
    import json

    from secrets_box import decrypt

    row, _ = seeded_client
    r = await client.put(
        f"/clients/{row.id}/bedrock-creds",
        json={
            "access_key_id": "AKIA-test",
            "secret_access_key": "sekrit",
            "region": "us-west-2",
        },
        headers=admin_headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["bedrock_creds_set_at"] is not None
    # Plaintext never echoed
    assert "secret_access_key" not in r.text
    assert "sekrit" not in r.text

    await db_session.refresh(row)
    assert row.bedrock_creds_encrypted is not None
    creds = json.loads(decrypt(row.bedrock_creds_encrypted))
    assert creds == {
        "access_key_id": "AKIA-test",
        "secret_access_key": "sekrit",
        "region": "us-west-2",
    }


async def test_clear_bedrock_creds_nulls_columns(
    client, admin_headers, seeded_client, secrets_key, db_session: AsyncSession
) -> None:
    row, _ = seeded_client
    await client.put(
        f"/clients/{row.id}/bedrock-creds",
        json={"access_key_id": "x", "secret_access_key": "y", "region": "us-east-1"},
        headers=admin_headers,
    )
    r = await client.delete(
        f"/clients/{row.id}/bedrock-creds", headers=admin_headers
    )
    assert r.status_code == 200
    assert r.json()["bedrock_creds_set_at"] is None
    await db_session.refresh(row)
    assert row.bedrock_creds_encrypted is None
    assert row.bedrock_creds_set_at is None


async def test_set_bedrock_creds_no_master_key_is_503(
    client, admin_headers, seeded_client, monkeypatch
) -> None:
    import secrets_box

    monkeypatch.setenv("CONDUCT_SECRETS_KEY", "")
    get_settings.cache_clear()
    secrets_box._fernet.cache_clear()
    row, _ = seeded_client
    r = await client.put(
        f"/clients/{row.id}/bedrock-creds",
        json={"access_key_id": "x", "secret_access_key": "y", "region": "us-east-1"},
        headers=admin_headers,
    )
    assert r.status_code == 503


async def test_set_bedrock_creds_unauthenticated_is_403(
    client, seeded_client
) -> None:
    row, _ = seeded_client
    r = await client.put(
        f"/clients/{row.id}/bedrock-creds",
        json={"access_key_id": "x", "secret_access_key": "y", "region": "us-east-1"},
    )
    assert r.status_code == 403


async def test_set_bedrock_creds_nonexistent_client_is_404(
    client, admin_headers, secrets_key
) -> None:
    r = await client.put(
        "/clients/00000000-0000-0000-0000-000000000000/bedrock-creds",
        json={"access_key_id": "x", "secret_access_key": "y", "region": "us-east-1"},
        headers=admin_headers,
    )
    assert r.status_code == 404


async def test_list_clients_includes_bedrock_creds_set_at(
    client, admin_headers, seeded_client, secrets_key
) -> None:
    row, _ = seeded_client
    await client.put(
        f"/clients/{row.id}/bedrock-creds",
        json={"access_key_id": "x", "secret_access_key": "y", "region": "us-east-1"},
        headers=admin_headers,
    )
    r = await client.get("/clients", headers=admin_headers)
    me = next(c for c in r.json() if c["id"] == str(row.id))
    assert me["bedrock_creds_set_at"] is not None


async def test_set_bedrock_creds_with_bearer_token_persists(
    client, admin_headers, seeded_client, secrets_key, db_session: AsyncSession
) -> None:
    import json

    from secrets_box import decrypt

    row, _ = seeded_client
    r = await client.put(
        f"/clients/{row.id}/bedrock-creds",
        json={"bearer_token": "ABSK-conduct-test", "region": "us-east-1"},
        headers=admin_headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["bedrock_creds_set_at"] is not None
    # No plaintext echoed
    assert "ABSK-conduct-test" not in r.text

    await db_session.refresh(row)
    creds = json.loads(decrypt(row.bedrock_creds_encrypted))
    assert creds == {"bearer_token": "ABSK-conduct-test", "region": "us-east-1"}
    # Pair fields must not be sneakily present
    assert "access_key_id" not in creds
    assert "secret_access_key" not in creds


async def test_set_bedrock_creds_rejects_both_styles_simultaneously(
    client, admin_headers, seeded_client, secrets_key
) -> None:
    row, _ = seeded_client
    r = await client.put(
        f"/clients/{row.id}/bedrock-creds",
        json={
            "bearer_token": "ABSK-x",
            "access_key_id": "AKIA-x",
            "secret_access_key": "y",
            "region": "us-east-1",
        },
        headers=admin_headers,
    )
    assert r.status_code == 422
    assert "not both" in r.text


async def test_set_bedrock_creds_rejects_neither_style(
    client, admin_headers, seeded_client, secrets_key
) -> None:
    row, _ = seeded_client
    r = await client.put(
        f"/clients/{row.id}/bedrock-creds",
        json={"region": "us-east-1"},
        headers=admin_headers,
    )
    assert r.status_code == 422
