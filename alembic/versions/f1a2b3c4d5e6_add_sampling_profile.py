"""add sampling profile (RoutingRule.sampling, Job.sampling)

Revision ID: f1a2b3c4d5e6
Revises: e85bbb33c2ea
Create Date: 2026-06-08 14:30:00.000000

Per-task generation profile (deterministic / balanced / creative) that bundles
temperature + seed policy. Existing rules default to 'balanced' so behavior is
unchanged. Job.sampling is the nullable per-request override (NULL = use the
rule).
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: str | Sequence[str] | None = "e85bbb33c2ea"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # RoutingRule.sampling — the task's generation profile. 'balanced' default
    # keeps pre-existing rules at moderate-temperature / random-seed behavior.
    op.add_column(
        "routing_rules",
        sa.Column(
            "sampling",
            sa.String(length=20),
            nullable=False,
            server_default="balanced",
        ),
    )
    # Job.sampling — nullable per-request override. NULL means "use the rule's
    # profile"; a value pins this single job to deterministic/balanced/creative.
    op.add_column(
        "jobs",
        sa.Column("sampling", sa.String(length=20), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("jobs", "sampling")
    op.drop_column("routing_rules", "sampling")
