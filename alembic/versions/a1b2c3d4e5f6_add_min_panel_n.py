"""add RoutingRule.min_panel_n (panel-judge quorum floor)

Revision ID: a1b2c3d4e5f6
Revises: f1a2b3c4d5e6
Create Date: 2026-06-21 16:00:00.000000

Optional per-rule quorum floor for panel judges (#21). NULL = no floor (existing
behavior: only a zero-survivor jury fails). When set, a panel that scores fewer
than min_panel_n jurors fails loudly rather than writing a degraded low-n
"median". Nullable, no server default, so existing rules are unchanged.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "routing_rules",
        sa.Column("min_panel_n", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("routing_rules", "min_panel_n")
