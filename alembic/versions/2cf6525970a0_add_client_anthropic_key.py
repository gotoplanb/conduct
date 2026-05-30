"""add client anthropic key columns

Revision ID: 2cf6525970a0
Revises: 0db8cc7e3e67
Create Date: 2026-05-30 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2cf6525970a0"
down_revision: str | Sequence[str] | None = "0db8cc7e3e67"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "client_apps",
        sa.Column("anthropic_api_key_encrypted", sa.Text(), nullable=True),
    )
    op.add_column(
        "client_apps",
        sa.Column("anthropic_api_key_set_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("client_apps", "anthropic_api_key_set_at")
    op.drop_column("client_apps", "anthropic_api_key_encrypted")
