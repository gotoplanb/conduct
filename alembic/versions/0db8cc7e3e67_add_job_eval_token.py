"""add job eval token

Revision ID: 0db8cc7e3e67
Revises: 366bdd65a54b
Create Date: 2026-05-25 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0db8cc7e3e67"
down_revision: str | Sequence[str] | None = "366bdd65a54b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("jobs", sa.Column("eval_token_hash", sa.String(length=64), nullable=True))
    op.add_column(
        "jobs", sa.Column("eval_token_expires_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "jobs",
        sa.Column(
            "eval_token_used", sa.Boolean(), server_default="false", nullable=False
        ),
    )
    op.create_index(
        op.f("ix_jobs_eval_token_hash"), "jobs", ["eval_token_hash"], unique=True
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_jobs_eval_token_hash"), table_name="jobs")
    op.drop_column("jobs", "eval_token_used")
    op.drop_column("jobs", "eval_token_expires_at")
    op.drop_column("jobs", "eval_token_hash")
