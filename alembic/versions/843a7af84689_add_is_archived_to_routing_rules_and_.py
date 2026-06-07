"""add is_archived to routing_rules and prompts

Revision ID: 843a7af84689
Revises: 8f3b900ba54d
Create Date: 2026-06-06 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "843a7af84689"
down_revision: str | Sequence[str] | None = "8f3b900ba54d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # routing_rules + prompts get a soft-delete flag. Existing rows default
    # to is_archived=false so nothing changes for live traffic. Indexed
    # because list-endpoints filter on it on every call.
    op.add_column(
        "routing_rules",
        sa.Column(
            "is_archived",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_index(
        "ix_routing_rules_is_archived",
        "routing_rules",
        ["is_archived"],
    )
    op.add_column(
        "prompts",
        sa.Column(
            "is_archived",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_index(
        "ix_prompts_is_archived",
        "prompts",
        ["is_archived"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_prompts_is_archived", table_name="prompts")
    op.drop_column("prompts", "is_archived")
    op.drop_index("ix_routing_rules_is_archived", table_name="routing_rules")
    op.drop_column("routing_rules", "is_archived")
