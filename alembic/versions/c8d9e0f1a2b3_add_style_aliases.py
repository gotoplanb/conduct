"""add style_aliases (logical style registry, #53)

Revision ID: c8d9e0f1a2b3
Revises: b7c8d9e0f1a2
Create Date: 2026-08-17 12:00:00.000000

Image sibling of voice_aliases: logical style name -> ComfyUI workflow +
params, client-scoped with shared defaults. Wander carries style names,
never workflow details — same boundary #50/#51 drew for voices.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8d9e0f1a2b3"
down_revision: str | Sequence[str] | None = "b7c8d9e0f1a2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "style_aliases",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column(
            "client_id",
            PG_UUID(as_uuid=True),
            sa.ForeignKey("client_apps.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("workflow_template", sa.String(200), nullable=False),
        sa.Column("params", JSONB(), nullable=False, server_default="{}"),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_archived", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.create_index("ix_style_aliases_name", "style_aliases", ["name"])
    op.create_index("ix_style_aliases_client_id", "style_aliases", ["client_id"])
    op.create_index(
        "ix_style_aliases_shared_unique",
        "style_aliases",
        ["name"],
        unique=True,
        postgresql_where=sa.text("client_id IS NULL"),
    )
    op.create_index(
        "ix_style_aliases_client_unique",
        "style_aliases",
        ["name", "client_id"],
        unique=True,
        postgresql_where=sa.text("client_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_table("style_aliases")
