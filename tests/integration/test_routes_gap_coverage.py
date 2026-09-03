"""Gap coverage for routes/models.py, routes/oauth.py, routes/tts.py,
routes/image.py, and routes/clients.py — mostly error paths and the
less-common parameter branches."""

from __future__ import annotations

import base64
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select

from config.settings import get_settings
from models.job import Job
from models.oauth import OAuthClient
from models.types import JobStatus
from oauth_provider import hash_secret, new_client_id, new_client_secret
from providers.base import ProviderError

REDIRECT = "https://claude.ai/api/mcp/auth_callback"


class _RaisingQueue:
    def enqueue(self, *args, **kwargs):
        raise RuntimeError("redis is down")


# --- /models ---------------------------------------------------------------


async def test_models_503_when_ollama_not_registered(app, client, admin_headers) -> None:
    from deps import get_provider_registry
    from providers.registry import ProviderRegistry

    app.dependency_overrides[get_provider_registry] = lambda: ProviderRegistry()
    r = await client.get("/models", headers=admin_headers)
    assert r.status_code == 503
    assert "ollama" in r.json()["detail"]


async def test_models_ollama_down_returns_empty_local(
    client, admin_headers, stub_registry
) -> None:
    async def _down():
        raise RuntimeError("connection refused")

    stub_registry.get("ollama").list_models = _down
    r = await client.get("/models", headers=admin_headers)
    assert r.status_code == 200
    assert r.json()["local"] == []


async def test_models_tolerates_bad_modified_at(client, admin_headers, stub_registry) -> None:
    async def _tags():
        return [
            {"name": "a:1b", "size": 0, "modified_at": "not-a-date"},
            {"name": "b:1b"},  # no modified_at at all
        ]

    stub_registry.get("ollama").list_models = _tags
    r = await client.get("/models", headers=admin_headers)
    assert r.status_code == 200
    by_name = {m["name"]: m for m in r.json()["local"]}
    assert by_name["a:1b"]["last_used"] is None
    assert by_name["a:1b"]["size_gb"] is None
    assert by_name["b:1b"]["last_used"] is None


async def test_load_model_provider_error_502(client, admin_headers, stub_registry) -> None:
    async def _fail(model, keep_alive=None):
        raise ProviderError("ollama says no")

    stub_registry.get("ollama").load = _fail
    r = await client.post("/models/a:1b/load", headers=admin_headers)
    assert r.status_code == 502
    assert "load failed" in r.json()["detail"]


async def test_load_model_unexpected_error_502(client, admin_headers, stub_registry) -> None:
    async def _fail(model, keep_alive=None):
        raise RuntimeError("socket closed")

    stub_registry.get("ollama").load = _fail
    r = await client.post("/models/a:1b/load", headers=admin_headers)
    assert r.status_code == 502
    assert "socket closed" in r.json()["detail"]


async def test_unload_model_provider_error_502(client, admin_headers, stub_registry) -> None:
    async def _fail(model):
        raise ProviderError("ollama says no")

    stub_registry.get("ollama").unload = _fail
    r = await client.post("/models/zzz:1b/unload", headers=admin_headers)
    assert r.status_code == 502
    assert "unload failed" in r.json()["detail"]


async def test_unload_model_unexpected_error_502(client, admin_headers, stub_registry) -> None:
    async def _fail(model):
        raise RuntimeError("socket closed")

    stub_registry.get("ollama").unload = _fail
    r = await client.post("/models/zzz:1b/unload", headers=admin_headers)
    assert r.status_code == 502
    assert "socket closed" in r.json()["detail"]


# --- /oauth ----------------------------------------------------------------


@pytest_asyncio.fixture
async def oauth_client(db_session, seeded_client):
    capp, _ = seeded_client
    raw_secret = new_client_secret()
    oc = OAuthClient(
        client_id=new_client_id(),
        client_secret_hash=hash_secret(raw_secret),
        name="gap-connector",
        client_app_id=capp.id,
        redirect_uris=[REDIRECT],
    )
    db_session.add(oc)
    await db_session.commit()
    await db_session.refresh(oc)
    return oc, raw_secret


async def test_authorize_wrong_response_type_redirects_error(client, oauth_client) -> None:
    oc, _ = oauth_client
    r = await client.get(
        "/oauth/authorize",
        params={
            "response_type": "token",
            "client_id": oc.client_id,
            "redirect_uri": REDIRECT,
            "code_challenge": "x",
            "state": "s1",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    qs = parse_qs(urlparse(r.headers["location"]).query)
    assert qs["error"] == ["unsupported_response_type"]
    assert qs["state"] == ["s1"]


async def test_authorize_missing_pkce_redirects_error(client, oauth_client) -> None:
    oc, _ = oauth_client
    r = await client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": oc.client_id,
            "redirect_uri": REDIRECT,
            "code_challenge": "",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    qs = parse_qs(urlparse(r.headers["location"]).query)
    assert qs["error"] == ["invalid_request"]


async def test_authorize_submit_without_admin_is_error_page(client, oauth_client) -> None:
    oc, _ = oauth_client
    r = await client.post(
        "/oauth/authorize",
        data={
            "client_id": oc.client_id,
            "redirect_uri": REDIRECT,
            "code_challenge": "x",
            "decision": "approve",
        },
    )
    assert r.status_code == 400
    assert "Admin session required" in r.text


async def test_authorize_submit_unknown_client_is_error_page(admin_client) -> None:
    r = await admin_client.post(
        "/oauth/authorize",
        data={
            "client_id": "cdtc_nope",
            "redirect_uri": REDIRECT,
            "code_challenge": "x",
            "decision": "approve",
        },
    )
    assert r.status_code == 400
    assert "Unknown client_id" in r.text


async def test_token_accepts_http_basic_client_auth(client, oauth_client) -> None:
    """Credentials in the Basic header authenticate the client: a bogus
    refresh token then fails as invalid_grant, not invalid_client."""
    oc, secret = oauth_client
    basic = base64.b64encode(f"{oc.client_id}:{secret}".encode()).decode("ascii")
    r = await client.post(
        "/oauth/token",
        data={"grant_type": "refresh_token", "refresh_token": "cdt_rt_bogus"},
        headers={"Authorization": f"Basic {basic}"},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_grant"


async def test_token_undecodable_basic_header_is_invalid_client(client, oauth_client) -> None:
    # Valid base64 of non-UTF-8 bytes — the decode raises, creds fall to "".
    basic = base64.b64encode(b"\xff\xfe").decode("ascii")
    r = await client.post(
        "/oauth/token",
        data={"grant_type": "refresh_token", "refresh_token": "x"},
        headers={"Authorization": f"Basic {basic}"},
    )
    assert r.status_code == 401
    assert r.json()["error"] == "invalid_client"


# --- /tts + /output --------------------------------------------------------


async def test_tts_enqueue_failure_flips_job_to_failed_503(
    client, db_session, seeded_client, fake_redis, monkeypatch: pytest.MonkeyPatch
) -> None:
    import routes.tts as tts_route

    monkeypatch.setattr(tts_route, "get_queue", _RaisingQueue)
    marker = f"tts gap {uuid4().hex[:8]}"
    r = await client.post(
        "/tts",
        json={"text": marker},
        headers={"Authorization": f"Bearer {seeded_client[1]}"},
    )
    assert r.status_code == 503
    assert "queue backend unavailable" in r.json()["detail"]
    job = await db_session.scalar(select(Job).where(Job.prompt == marker))
    assert job.status == JobStatus.FAILED.value


def _output_dir() -> Path:
    d = Path(get_settings().tts_output_dir).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


async def test_output_serves_existing_file(client, admin_headers) -> None:
    path = _output_dir() / f"{uuid4()}.mp3"
    path.write_bytes(b"ID3 fake mp3 bytes")
    try:
        r = await client.get(f"/output/{path.name}", headers=admin_headers)
        assert r.status_code == 200
        assert r.headers["content-type"] == "audio/mpeg"
        assert r.content == b"ID3 fake mp3 bytes"
    finally:
        path.unlink()


async def test_output_symlink_escaping_output_dir_400(
    client, admin_headers, tmp_path
) -> None:
    """A UUID-named symlink pointing outside the output dir must be refused —
    the resolved path check is the backstop behind the filename regex."""
    outside = tmp_path / "secret.txt"
    outside.write_text("private")
    link = _output_dir() / f"{uuid4()}.mp3"
    link.symlink_to(outside)
    try:
        r = await client.get(f"/output/{link.name}", headers=admin_headers)
        assert r.status_code == 400
        assert r.json()["detail"] == "invalid path"
    finally:
        link.unlink()


# --- /image + /styles/registry ---------------------------------------------


async def test_image_enqueue_failure_flips_job_to_failed_503(
    client, db_session, seeded_client, fake_redis, monkeypatch: pytest.MonkeyPatch
) -> None:
    import worker.queue as wq

    monkeypatch.setattr(wq, "get_media_queue", _RaisingQueue)
    marker = f"image gap {uuid4().hex[:8]}"
    r = await client.post(
        "/image",
        json={"prompt": marker},
        headers={"Authorization": f"Bearer {seeded_client[1]}"},
    )
    assert r.status_code == 503
    assert "queue backend unavailable" in r.json()["detail"]
    job = await db_session.scalar(select(Job).where(Job.prompt == marker))
    assert job.status == JobStatus.FAILED.value


async def test_styles_registry_upsert_list_archive_roundtrip(
    client, admin_headers
) -> None:
    name = f"gap-style-{uuid4().hex[:8]}"
    r = await client.put(
        f"/styles/registry/{name}",
        json={"workflow_template": "wander_scene_image", "params": {"width": 1024}},
        headers=admin_headers,
    )
    assert r.status_code == 200
    assert r.json()["is_archived"] is False

    listed = await client.get("/styles/registry", headers=admin_headers)
    assert name in [a["name"] for a in listed.json()["aliases"]]

    archived = await client.delete(f"/styles/registry/{name}", headers=admin_headers)
    assert archived.status_code == 200
    assert archived.json()["is_archived"] is True

    # Archived aliases drop out of the default listing but show with the flag.
    listed = await client.get("/styles/registry", headers=admin_headers)
    assert name not in [a["name"] for a in listed.json()["aliases"]]
    listed = await client.get(
        "/styles/registry", params={"include_archived": "true"}, headers=admin_headers
    )
    assert name in [a["name"] for a in listed.json()["aliases"]]


async def test_styles_registry_name_too_long_400(client, admin_headers) -> None:
    r = await client.put(
        f"/styles/registry/{'x' * 101}",
        json={"workflow_template": "wander_scene_image"},
        headers=admin_headers,
    )
    assert r.status_code == 400
    assert "1-100 chars" in r.json()["detail"]


async def test_styles_registry_archive_unknown_404(client, admin_headers) -> None:
    r = await client.delete("/styles/registry/never-existed", headers=admin_headers)
    assert r.status_code == 404


# --- /clients --------------------------------------------------------------


async def test_create_client_conflict_409(
    client, admin_headers, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The create handler maps IntegrityError to 409. The live schema has no
    unique constraint on client_apps.name, so the conflict is simulated at the
    session boundary."""
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
    r = await client.post("/clients", json={"name": "dupe-client"}, headers=admin_headers)
    assert r.status_code == 409
    assert "conflict" in r.json()["detail"]


async def test_clear_anthropic_key_unknown_client_404(client, admin_headers) -> None:
    r = await client.delete(f"/clients/{uuid4()}/anthropic-key", headers=admin_headers)
    assert r.status_code == 404


async def test_clear_bedrock_creds_unknown_client_404(client, admin_headers) -> None:
    r = await client.delete(f"/clients/{uuid4()}/bedrock-creds", headers=admin_headers)
    assert r.status_code == 404
