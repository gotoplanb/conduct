"""add per-client ElevenLabs API key to client_apps (#52)

Revision ID: e7f8a9b0c1d2
Revises: d4e5f6a7b8c9
Create Date: 2026-11-18 12:00:00.000000

Mirrors anthropic_api_key_encrypted: a Fernet ciphertext plus a set-at
timestamp. Null means the client cannot use ElevenLabs voices.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e7f8a9b0c1d2"
down_revision: str | Sequence[str] | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "client_apps", sa.Column("elevenlabs_api_key_encrypted", sa.Text(), nullable=True)
    )
    op.add_column(
        "client_apps",
        sa.Column("elevenlabs_api_key_set_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("client_apps", "elevenlabs_api_key_set_at")
    op.drop_column("client_apps", "elevenlabs_api_key_encrypted")
