"""Routes under /tts and /output."""

from __future__ import annotations

from uuid import uuid4


async def test_submit_tts_returns_202(client, db_session, seeded_client, fake_redis) -> None:
    r = await client.post(
        "/tts",
        json={"text": "hello world"},
        headers={"Authorization": f"Bearer {seeded_client[1]}"},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "pending"
    assert "expected_output_url" in body


async def test_submit_tts_oversized_returns_413(client, seeded_client, fake_redis) -> None:
    r = await client.post(
        "/tts",
        json={"text": "a" * 20000},
        headers={"Authorization": f"Bearer {seeded_client[1]}"},
    )
    assert r.status_code == 413


async def test_submit_tts_requires_client_auth(client) -> None:
    r = await client.post("/tts", json={"text": "x"})
    assert r.status_code == 403


async def test_output_rejects_non_uuid_filename(client, admin_headers) -> None:
    r = await client.get("/output/audiobook.mp3", headers=admin_headers)
    assert r.status_code == 400


async def test_output_returns_404_for_missing_file(client, admin_headers) -> None:
    r = await client.get(f"/output/{uuid4()}.mp3", headers=admin_headers)
    assert r.status_code == 404


async def test_output_requires_admin(client) -> None:
    r = await client.get(f"/output/{uuid4()}.mp3")
    assert r.status_code == 403


async def test_output_accepts_cookie_auth(client, admin_token) -> None:
    """Browser-side audio player uses the UI session cookie, not the
    Authorization header. Both auth modes must work on /output."""
    cookies = {"conduct_admin": admin_token}
    r = await client.get(f"/output/{uuid4()}.mp3", cookies=cookies)
    # 404 (file doesn't exist) is fine — proves we got past the auth gate
    assert r.status_code == 404
