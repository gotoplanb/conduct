"""add key_created_at to client_apps

Revision ID: e892267dc2e0
Revises: 0da6d7df2008
Create Date: 2026-05-21 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e892267dc2e0"
down_revision: str | Sequence[str] | None = "0da6d7df2008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add nullable first so we can backfill, then enforce NOT NULL.
    op.add_column(
        "client_apps",
        sa.Column("key_created_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Existing keys date back to when the client was created.
    op.execute("UPDATE client_apps SET key_created_at = created_at WHERE key_created_at IS NULL")
    op.alter_column("client_apps", "key_created_at", nullable=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("client_apps", "key_created_at")
