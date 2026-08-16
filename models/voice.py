from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class VoiceAlias(Base):
    """A named voice: logical name -> concrete synthesis config (#51).

    `client_id IS NULL` is the shared/default entry; a non-NULL client_id is a
    per-client override that wins at resolution time — same scoping shape as
    Prompt. Clients submit TTS with the logical name and never carry engine
    or voice-file details; Conduct owns the files and validates existence.
    """

    __tablename__ = "voice_aliases"
    __table_args__ = (
        Index(
            "ix_voice_aliases_shared_unique",
            "name",
            unique=True,
            postgresql_where="client_id IS NULL",
        ),
        Index(
            "ix_voice_aliases_client_unique",
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
    # Synthesis engine. Only 'piper' exists today; the field is here so a
    # cloud engine (e.g. elevenlabs) is a new row value, not a schema change.
    engine: Mapped[str] = mapped_column(
        String(20), nullable=False, default="piper", server_default="piper"
    )
    # Engine-specific voice identifier. For piper: the voice file stem in
    # tts_voices_dir (e.g. 'en_US-amy-medium' -> en_US-amy-medium.onnx).
    voice_file: Mapped[str] = mapped_column(String(200), nullable=False)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    # Soft-delete flag, same rationale as RoutingRule/Prompt: archived rows
    # are invisible to resolution and listings but keep history readable.
    is_archived: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
