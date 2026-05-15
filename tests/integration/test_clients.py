"""Routes under /clients — admin CRUD + per-client usage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

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


async def test_duplicate_name_is_currently_allowed(client, admin_headers) -> None:
    """Documents current behavior: ClientApp.name has no unique constraint.
    The route catches IntegrityError on api_key_hash only (which collides
    with negligible probability). If a name-unique constraint is added
    later, flip this test to expect 409."""
    await client.post("/clients", json={"name": "dup"}, headers=admin_headers)
    r = await client.post("/clients", json={"name": "dup"}, headers=admin_headers)
    assert r.status_code == 201


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
