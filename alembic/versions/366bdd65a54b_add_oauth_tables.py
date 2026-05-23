"""add oauth tables

Revision ID: 366bdd65a54b
Revises: e892267dc2e0
Create Date: 2026-05-23 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "366bdd65a54b"
down_revision: str | Sequence[str] | None = "e892267dc2e0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "oauth_clients",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("client_id", sa.String(length=64), nullable=False),
        sa.Column("client_secret_hash", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("client_app_id", sa.UUID(), nullable=False),
        sa.Column("redirect_uris", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(length=64), server_default="", nullable=False),
        sa.ForeignKeyConstraint(["client_app_id"], ["client_apps.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_oauth_clients_client_app_id"), "oauth_clients", ["client_app_id"], unique=False
    )
    op.create_index(
        op.f("ix_oauth_clients_client_id"), "oauth_clients", ["client_id"], unique=True
    )

    op.create_table(
        "oauth_authorization_codes",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("code_hash", sa.String(length=64), nullable=False),
        sa.Column("client_id", sa.String(length=64), nullable=False),
        sa.Column("client_app_id", sa.UUID(), nullable=False),
        sa.Column("redirect_uri", sa.Text(), nullable=False),
        sa.Column("code_challenge", sa.String(length=128), nullable=False),
        sa.Column("code_challenge_method", sa.String(length=10), nullable=False),
        sa.Column("scope", sa.String(length=255), server_default="", nullable=False),
        sa.Column("used", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_oauth_authorization_codes_client_id"),
        "oauth_authorization_codes",
        ["client_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_oauth_authorization_codes_code_hash"),
        "oauth_authorization_codes",
        ["code_hash"],
        unique=True,
    )

    op.create_table(
        "oauth_tokens",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("access_token_hash", sa.String(length=64), nullable=False),
        sa.Column("refresh_token_hash", sa.String(length=64), nullable=True),
        sa.Column("client_id", sa.String(length=64), nullable=False),
        sa.Column("client_app_id", sa.UUID(), nullable=False),
        sa.Column("scope", sa.String(length=255), server_default="", nullable=False),
        sa.Column("revoked", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("access_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("refresh_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_oauth_tokens_access_token_hash"),
        "oauth_tokens",
        ["access_token_hash"],
        unique=True,
    )
    op.create_index(
        op.f("ix_oauth_tokens_client_app_id"), "oauth_tokens", ["client_app_id"], unique=False
    )
    op.create_index(
        op.f("ix_oauth_tokens_client_id"), "oauth_tokens", ["client_id"], unique=False
    )
    op.create_index(
        op.f("ix_oauth_tokens_refresh_token_hash"),
        "oauth_tokens",
        ["refresh_token_hash"],
        unique=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_oauth_tokens_refresh_token_hash"), table_name="oauth_tokens")
    op.drop_index(op.f("ix_oauth_tokens_client_id"), table_name="oauth_tokens")
    op.drop_index(op.f("ix_oauth_tokens_client_app_id"), table_name="oauth_tokens")
    op.drop_index(op.f("ix_oauth_tokens_access_token_hash"), table_name="oauth_tokens")
    op.drop_table("oauth_tokens")
    op.drop_index(
        op.f("ix_oauth_authorization_codes_code_hash"), table_name="oauth_authorization_codes"
    )
    op.drop_index(
        op.f("ix_oauth_authorization_codes_client_id"), table_name="oauth_authorization_codes"
    )
    op.drop_table("oauth_authorization_codes")
    op.drop_index(op.f("ix_oauth_clients_client_id"), table_name="oauth_clients")
    op.drop_index(op.f("ix_oauth_clients_client_app_id"), table_name="oauth_clients")
    op.drop_table("oauth_clients")
