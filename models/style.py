from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class StyleAlias(Base):
    """A named visual style: logical name -> ComfyUI workflow + params (#53).

    The image sibling of VoiceAlias: clients (Wander) submit /image jobs with
    a logical style name (`backdrop-wide`) and never carry workflow files,
    checkpoints, or dimensions. Same scoping as prompts/voices — client_id
    NULL is the shared default, non-NULL is a per-client override.
    """

    __tablename__ = "style_aliases"
    __table_args__ = (
        Index(
            "ix_style_aliases_shared_unique",
            "name",
            unique=True,
            postgresql_where="client_id IS NULL",
        ),
        Index(
            "ix_style_aliases_client_unique",
            "name",
            "client_id",
            unique=True,
            postgresql_where="client_id IS NOT NULL",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    client_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("client_apps.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    # ComfyUI workflow template name (a JSON file in comfy_workflows/).
    workflow_template: Mapped[str] = mapped_column(String(200), nullable=False)
    # Per-style parameter overrides injected via the template's _meta.inject
    # map (width, height, seed, negative_prompt, ...). Merged over the
    # template's default_params at generation time.
    params: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    is_archived: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
