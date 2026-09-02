"""Conduct's answers to the fleet's OAuth contract.

The contract and its assertions come from `oauth-conformance`, shared with the
other authorization servers in the fleet. This file is only the adapter — this
service's answers to questions asked identically of every implementation.

**Conduct's principal is a machine.** `ClientApp` is a pure machine identity
with no owner, and `OAuthClient` has a foreign key to exactly one of them — so
a client here cannot act for two principals however hard you push. That makes
deriving the principal from the client record correct, where the same line was
a defect in crossover once its principal became a human and the relationship
became one-to-many.
"""

from __future__ import annotations

import secrets
from uuid import uuid4

import pytest
from oauth_conformance import Client, Principal, Tokens
from oauth_conformance.suite import OAuthConformanceSuite
from sqlalchemy import select

from models.client import ClientApp
from models.oauth import OAuthClient
from oauth_provider import (
    DEFAULT_SCOPE,
    hash_secret,
    issue_authorization_code,
    new_client_id,
    new_client_secret,
    redeem_authorization_code,
    refresh_token_grant,
    resolve_access_token,
)


class ConductOAuth:
    # One client, one ClientApp, enforced by a foreign key. The suite skips the
    # assertions that need two principals per client and runs a narrower one.
    separates_client_from_principal = False

    def __init__(self, session):
        self._session = session

    async def make_principal(self, kind: str = "machine") -> Principal:
        app = ClientApp(name=f"conformance-{uuid4().hex[:8]}", api_key_hash=secrets.token_hex(32))
        self._session.add(app)
        await self._session.commit()
        await self._session.refresh(app)
        return Principal("machine", app.id)

    async def register_client(
        self, *, redirect_uris: list[str], registrant: Principal | None = None
    ) -> Client:
        owner = registrant or await self.make_principal()
        secret = new_client_secret()
        row = OAuthClient(
            client_id=new_client_id(),
            client_secret_hash=hash_secret(secret),
            name="conformance",
            client_app_id=owner.id,
            redirect_uris=list(redirect_uris),
        )
        self._session.add(row)
        await self._session.commit()
        await self._session.refresh(row)
        return Client(
            id=row.client_id, secret=secret, redirect_uris=list(redirect_uris), registrant=owner
        )

    async def _row(self, client: Client) -> OAuthClient:
        row = await self._session.scalar(
            select(OAuthClient).where(OAuthClient.client_id == client.id)
        )
        assert row is not None
        return row

    async def authorize(
        self,
        *,
        client: Client,
        principal: Principal,
        redirect_uri: str,
        code_challenge: str,
        code_challenge_method: str = "S256",
    ) -> str:
        row = await self._row(client)
        # Conduct takes no principal: it reads `client.client_app_id`. Sound
        # here precisely because the two cannot differ, which this asserts
        # rather than assumes.
        assert Principal("machine", row.client_app_id) == principal, (
            "conduct cannot issue a grant for a principal other than the client's own"
        )
        return await issue_authorization_code(
            self._session,
            client=row,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            scope=DEFAULT_SCOPE,
        )

    async def exchange(
        self, *, client: Client, code: str, verifier: str, redirect_uri: str
    ) -> Tokens:
        token = await redeem_authorization_code(
            self._session,
            client=await self._row(client),
            code=code,
            redirect_uri=redirect_uri,
            code_verifier=verifier,
        )
        return Tokens(access=token.raw_access_token, refresh=token.raw_refresh_token)

    async def refresh(self, *, client: Client, refresh_token: str) -> Tokens:
        token = await refresh_token_grant(
            self._session, client=await self._row(client), refresh_token=refresh_token
        )
        return Tokens(access=token.raw_access_token, refresh=token.raw_refresh_token)

    async def resolve(self, access_token: str) -> Principal | None:
        app = await resolve_access_token(self._session, access_token)
        return Principal("machine", app.id) if app else None

    async def deactivate_client(self, client: Client) -> None:
        row = await self._row(client)
        row.is_active = False
        await self._session.commit()


@pytest.fixture
async def oauth(db_session):
    return ConductOAuth(db_session)


class TestConductConforms(OAuthConformanceSuite):
    """Every shared assertion, against conduct."""
