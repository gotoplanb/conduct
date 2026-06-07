from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class Prompt(Base):
    """A task-type prompt. `client_id IS NULL` is the shared/default;
    `client_id IS NOT NULL` is a per-client override that wins at resolution
    time. Uniqueness is enforced by two partial indexes (Postgres' way of
    saying "unique unless NULL")."""

    __tablename__ = "prompts"
    __table_args__ = (
        Index(
            "ix_prompts_shared_unique",
            "task_type",
            unique=True,
            postgresql_where="client_id IS NULL",
        ),
        Index(
            "ix_prompts_client_unique",
            "task_type",
            "client_id",
            unique=True,
            postgresql_where="client_id IS NOT NULL",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    client_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("client_apps.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    updated_by: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    # Soft-delete flag — DELETE /prompts/{task_type}?client=name flips this
    # rather than dropping the row, so PromptVersion history stays linked
    # and historic jobs render correctly. resolve_prompt() ignores archived
    # rows (raises PromptNotFoundError if there's no live row to use).
    is_archived: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )


class PromptVersion(Base):
    """Append-only history of every Prompt save. Job rows reference this
    table by id (Job.metadata.prompt.version_id) so each job can be traced
    back to the exact prompt content it ran against, even after the
    canonical row has been edited."""

    __tablename__ = "prompt_versions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    client_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("client_apps.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    edited_by: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    edited_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )
