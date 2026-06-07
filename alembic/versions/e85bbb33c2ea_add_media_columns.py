"""add media columns (Job.inputs, Job.media_url, RoutingRule.media_kind)

Revision ID: e85bbb33c2ea
Revises: 843a7af84689
Create Date: 2026-06-07 11:30:00.000000

Foundation for the media-task primitives (text→image, text→audio,
image→video, video+audio→mux). Existing text rules default to
media_kind='text' so the migration is fully backward compatible.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e85bbb33c2ea"
down_revision: str | Sequence[str] | None = "843a7af84689"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Job.inputs — typed-bag of per-task-type input fields (e.g. source_image_url,
    # source_video_url, source_audio_url). Text-only tasks leave this {}.
    op.add_column(
        "jobs",
        sa.Column(
            "inputs",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    # Job.media_url — set by media tasks on completion. Mirrors how the TTS
    # executor already returns a URL (/output/{job.id}.mp3); media tasks
    # extend the same pattern to images, videos, and mux'd composites.
    op.add_column(
        "jobs",
        sa.Column("media_url", sa.Text(), nullable=True),
    )
    # RoutingRule.media_kind — declares the task's output shape. The worker
    # branches dispatch on this: 'text' uses the existing BaseProvider path;
    # everything else uses BaseMediaProvider via the new conduct-media queue.
    op.add_column(
        "routing_rules",
        sa.Column(
            "media_kind",
            sa.String(length=16),
            nullable=False,
            server_default="text",
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("routing_rules", "media_kind")
    op.drop_column("jobs", "media_url")
    op.drop_column("jobs", "inputs")
