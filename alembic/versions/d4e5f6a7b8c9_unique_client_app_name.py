"""unique constraint on client_apps.name

Revision ID: d4e5f6a7b8c9
Revises: c8d9e0f1a2b3
Create Date: 2026-09-03 19:20:00.000000

The duplicate-name IntegrityError handlers in routes/clients.py and
routes/ui.py have existed since the client CRUD landed, but the constraint
they were written for never did — creating two clients with the same name
silently succeeded. Names are how operators identify clients everywhere
(routing rules, usage rollups, the UI), so duplicates are ambiguity, not a
feature. The live DB was verified duplicate-free before this migration.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: str | Sequence[str] | None = "c8d9e0f1a2b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_unique_constraint("uq_client_apps_name", "client_apps", ["name"])


def downgrade() -> None:
    op.drop_constraint("uq_client_apps_name", "client_apps", type_="unique")
