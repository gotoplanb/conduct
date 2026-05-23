"""OAuth 2.0 authorization-server tables.

Conduct acts as a minimal OAuth provider so external clients (the Claude
custom connector / iOS app) can connect over the MCP server with a
user-approved token instead of a raw admin key. Every OAuth client is bound
to a ClientApp, so jobs created through MCP inherit that client's
attribution, rate limits, and cloud permissions.

Secrets, codes, and tokens are only ever stored as SHA-256 hashes — the raw
values are shown once at creation/issue time and never again.
"""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class OAuthClient(Base):
    """A registered OAuth client (a 'connector', e.g. Dave's iOS app). Bound
    to a ClientApp for job attribution."""

    __tablename__ = "oauth_clients"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    client_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    client_secret_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    client_app_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("client_apps.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    redirect_uris: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    created_by: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )


class OAuthAuthorizationCode(Base):
    """A short-lived authorization code issued at /authorize and redeemed
    once at /token. PKCE challenge is stored and verified on redemption."""

    __tablename__ = "oauth_authorization_codes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    client_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    client_app_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    redirect_uri: Mapped[str] = mapped_column(Text, nullable=False)
    code_challenge: Mapped[str] = mapped_column(String(128), nullable=False)
    code_challenge_method: Mapped[str] = mapped_column(String(10), nullable=False, default="S256")
    scope: Mapped[str] = mapped_column(String(255), nullable=False, default="", server_default="")
    used: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class OAuthToken(Base):
    """An issued access token (with optional refresh token). Both are stored
    hashed. Revoked or expired rows are kept for audit but won't authenticate."""

    __tablename__ = "oauth_tokens"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    access_token_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    refresh_token_hash: Mapped[str | None] = mapped_column(
        String(64), unique=True, nullable=True, index=True
    )
    client_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    client_app_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False, index=True)
    scope: Mapped[str] = mapped_column(String(255), nullable=False, default="", server_default="")
    revoked: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    access_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    refresh_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
