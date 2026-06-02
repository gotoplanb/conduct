"""add client bedrock creds columns

Revision ID: 8f3b900ba54d
Revises: 2cf6525970a0
Create Date: 2026-06-02 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8f3b900ba54d"
down_revision: str | Sequence[str] | None = "2cf6525970a0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "client_apps",
        sa.Column("bedrock_creds_encrypted", sa.Text(), nullable=True),
    )
    op.add_column(
        "client_apps",
        sa.Column("bedrock_creds_set_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("client_apps", "bedrock_creds_set_at")
    op.drop_column("client_apps", "bedrock_creds_encrypted")
