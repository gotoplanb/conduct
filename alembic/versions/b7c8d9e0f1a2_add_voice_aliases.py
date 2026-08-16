"""add voice_aliases (named-voice registry, #51)

Revision ID: b7c8d9e0f1a2
Revises: a1b2c3d4e5f6
Create Date: 2026-08-16 15:00:00.000000

Logical voice name -> concrete synthesis config, client-scoped with shared
defaults (client_id IS NULL), mirroring the prompts table's partial-unique-
index scoping. Decided in #50: clients reference voices by name; Conduct
resolves and validates them.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7c8d9e0f1a2"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "voice_aliases",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column(
            "client_id",
            PG_UUID(as_uuid=True),
            sa.ForeignKey("client_apps.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("engine", sa.String(20), nullable=False, server_default="piper"),
        sa.Column("voice_file", sa.String(200), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_archived", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.create_index("ix_voice_aliases_name", "voice_aliases", ["name"])
    op.create_index("ix_voice_aliases_client_id", "voice_aliases", ["client_id"])
    op.create_index(
        "ix_voice_aliases_shared_unique",
        "voice_aliases",
        ["name"],
        unique=True,
        postgresql_where=sa.text("client_id IS NULL"),
    )
    op.create_index(
        "ix_voice_aliases_client_unique",
        "voice_aliases",
        ["name", "client_id"],
        unique=True,
        postgresql_where=sa.text("client_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_table("voice_aliases")
