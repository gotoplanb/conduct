"""OAuth 2.0 authorization server — discovery, auth-code + PKCE flow,
refresh, and the resource-server token resolver."""

from __future__ import annotations

import base64
import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import pytest
import pytest_asyncio
from sqlalchemy import select

from models.oauth import OAuthClient, OAuthToken
from oauth_provider import (
    hash_secret,
    new_client_id,
    new_client_secret,
    resolve_access_token,
    verify_pkce,
)

REDIRECT = "https://claude.ai/api/mcp/auth_callback"


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    return verifier, challenge


@pytest_asyncio.fixture
async def oauth_client(db_session, seeded_client):
    capp, _ = seeded_client
    raw_secret = new_client_secret()
    client = OAuthClient(
        client_id=new_client_id(),
        client_secret_hash=hash_secret(raw_secret),
        name="dave-ios",
        client_app_id=capp.id,
        redirect_uris=[REDIRECT],
    )
    db_session.add(client)
    await db_session.commit()
    await db_session.refresh(client)
    return client, raw_secret


async def _approve_and_get_code(client, oc, admin_token, challenge) -> str:
    r = await client.post(
        "/oauth/authorize",
        data={
            "client_id": oc.client_id,
            "redirect_uri": REDIRECT,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "scope": "mcp",
            "state": "xyz",
            "decision": "approve",
        },
        cookies={"conduct_admin": admin_token},
        follow_redirects=False,
    )
    assert r.status_code == 303
    loc = urlparse(r.headers["location"])
    qs = parse_qs(loc.query)
    assert qs["state"] == ["xyz"]
    return qs["code"][0]


# --- pure helpers ---


def test_verify_pkce_roundtrip() -> None:
    verifier, challenge = _pkce()
    assert verify_pkce(verifier, challenge, "S256")
    assert not verify_pkce("wrong", challenge, "S256")
    assert not verify_pkce(verifier, challenge, "plain")  # plain disallowed


# --- discovery ---


async def test_discovery_authorization_server(client) -> None:
    r = await client.get("/.well-known/oauth-authorization-server")
    assert r.status_code == 200
    body = r.json()
    assert body["authorization_endpoint"].endswith("/oauth/authorize")
    assert body["token_endpoint"].endswith("/oauth/token")
    assert body["code_challenge_methods_supported"] == ["S256"]


async def test_discovery_protected_resource(client) -> None:
    r = await client.get("/.well-known/oauth-protected-resource")
    assert r.status_code == 200
    assert r.json()["resource"].endswith("/mcp")


# --- authorize ---


async def test_authorize_without_login_redirects_to_login(client, oauth_client) -> None:
    oc, _ = oauth_client
    _, challenge = _pkce()
    r = await client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": oc.client_id,
            "redirect_uri": REDIRECT,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"].startswith("/ui/login?next=")


async def test_authorize_shows_consent_when_logged_in(client, oauth_client, admin_token) -> None:
    oc, _ = oauth_client
    _, challenge = _pkce()
    r = await client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": oc.client_id,
            "redirect_uri": REDIRECT,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        cookies={"conduct_admin": admin_token},
    )
    assert r.status_code == 200
    assert "Authorize connection" in r.text
    assert "dave-ios" in r.text


async def test_authorize_unknown_client_is_error_page(client, admin_token) -> None:
    _, challenge = _pkce()
    r = await client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": "cdtc_nope",
            "redirect_uri": REDIRECT,
            "code_challenge": challenge,
        },
        cookies={"conduct_admin": admin_token},
    )
    assert r.status_code == 400


async def test_authorize_bad_redirect_uri_is_error_page(client, oauth_client, admin_token) -> None:
    oc, _ = oauth_client
    _, challenge = _pkce()
    r = await client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": oc.client_id,
            "redirect_uri": "https://evil.example/cb",
            "code_challenge": challenge,
        },
        cookies={"conduct_admin": admin_token},
    )
    assert r.status_code == 400


async def test_authorize_deny_redirects_with_error(client, oauth_client, admin_token) -> None:
    oc, _ = oauth_client
    _, challenge = _pkce()
    r = await client.post(
        "/oauth/authorize",
        data={
            "client_id": oc.client_id,
            "redirect_uri": REDIRECT,
            "code_challenge": challenge,
            "decision": "deny",
            "state": "s1",
        },
        cookies={"conduct_admin": admin_token},
        follow_redirects=False,
    )
    assert r.status_code == 303
    qs = parse_qs(urlparse(r.headers["location"]).query)
    assert qs["error"] == ["access_denied"]


# --- token ---


async def test_full_auth_code_flow(client, oauth_client, admin_token) -> None:
    oc, secret = oauth_client
    verifier, challenge = _pkce()
    code = await _approve_and_get_code(client, oc, admin_token, challenge)

    r = await client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT,
            "code_verifier": verifier,
            "client_id": oc.client_id,
            "client_secret": secret,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "Bearer"
    assert body["access_token"].startswith("cdt_at_")
    assert body["refresh_token"].startswith("cdt_rt_")
    assert body["expires_in"] == 3600


async def test_code_is_single_use(client, oauth_client, admin_token) -> None:
    oc, secret = oauth_client
    verifier, challenge = _pkce()
    code = await _approve_and_get_code(client, oc, admin_token, challenge)
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT,
        "code_verifier": verifier,
        "client_id": oc.client_id,
        "client_secret": secret,
    }
    assert (await client.post("/oauth/token", data=data)).status_code == 200
    reused = await client.post("/oauth/token", data=data)
    assert reused.status_code == 400
    assert reused.json()["error"] == "invalid_grant"


async def test_token_wrong_pkce_verifier_rejected(client, oauth_client, admin_token) -> None:
    oc, secret = oauth_client
    _, challenge = _pkce()
    code = await _approve_and_get_code(client, oc, admin_token, challenge)
    r = await client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT,
            "code_verifier": "not-the-verifier",
            "client_id": oc.client_id,
            "client_secret": secret,
        },
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_grant"


async def test_token_bad_client_secret_is_401(client, oauth_client, admin_token) -> None:
    oc, _ = oauth_client
    verifier, challenge = _pkce()
    code = await _approve_and_get_code(client, oc, admin_token, challenge)
    r = await client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT,
            "code_verifier": verifier,
            "client_id": oc.client_id,
            "client_secret": "cdts_wrong",
        },
    )
    assert r.status_code == 401
    assert r.json()["error"] == "invalid_client"


async def test_refresh_token_grant_rotates(client, oauth_client, admin_token) -> None:
    oc, secret = oauth_client
    verifier, challenge = _pkce()
    code = await _approve_and_get_code(client, oc, admin_token, challenge)
    first = (
        await client.post(
            "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT,
                "code_verifier": verifier,
                "client_id": oc.client_id,
                "client_secret": secret,
            },
        )
    ).json()

    r = await client.post(
        "/oauth/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": first["refresh_token"],
            "client_id": oc.client_id,
            "client_secret": secret,
        },
    )
    assert r.status_code == 200
    new = r.json()
    assert new["access_token"] != first["access_token"]

    # The old refresh token is now revoked.
    reused = await client.post(
        "/oauth/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": first["refresh_token"],
            "client_id": oc.client_id,
            "client_secret": secret,
        },
    )
    assert reused.status_code == 400


async def test_unsupported_grant_type(client, oauth_client) -> None:
    oc, secret = oauth_client
    r = await client.post(
        "/oauth/token",
        data={"grant_type": "password", "client_id": oc.client_id, "client_secret": secret},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "unsupported_grant_type"


# --- resource-server token resolution ---


async def test_resolve_access_token(db_session, oauth_client, client, admin_token) -> None:
    oc, secret = oauth_client
    verifier, challenge = _pkce()
    code = await _approve_and_get_code(client, oc, admin_token, challenge)
    body = (
        await client.post(
            "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT,
                "code_verifier": verifier,
                "client_id": oc.client_id,
                "client_secret": secret,
            },
        )
    ).json()

    app = await resolve_access_token(db_session, body["access_token"])
    assert app is not None
    assert app.id == oc.client_app_id

    assert await resolve_access_token(db_session, "cdt_at_garbage") is None


async def test_resolve_expired_token_returns_none(db_session, oauth_client) -> None:
    oc, _ = oauth_client
    # Forge an already-expired token row directly.
    raw = "cdt_at_expired_example"
    db_session.add(
        OAuthToken(
            access_token_hash=hash_secret(raw),
            client_id=oc.client_id,
            client_app_id=oc.client_app_id,
            scope="mcp",
            access_expires_at=datetime.now(UTC) - timedelta(minutes=1),
        )
    )
    await db_session.commit()
    assert await resolve_access_token(db_session, raw) is None


async def test_resolve_revoked_token_returns_none(db_session, oauth_client) -> None:
    oc, _ = oauth_client
    raw = "cdt_at_revoked_example"
    db_session.add(
        OAuthToken(
            access_token_hash=hash_secret(raw),
            client_id=oc.client_id,
            client_app_id=oc.client_app_id,
            scope="mcp",
            revoked=True,
            access_expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
    )
    await db_session.commit()
    assert await resolve_access_token(db_session, raw) is None


@pytest.mark.usefixtures("oauth_client")
async def test_token_rows_persisted(db_session, oauth_client, client, admin_token) -> None:
    oc, secret = oauth_client
    verifier, challenge = _pkce()
    code = await _approve_and_get_code(client, oc, admin_token, challenge)
    await client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT,
            "code_verifier": verifier,
            "client_id": oc.client_id,
            "client_secret": secret,
        },
    )
    rows = (
        await db_session.scalars(select(OAuthToken).where(OAuthToken.client_id == oc.client_id))
    ).all()
    assert len(rows) == 1
