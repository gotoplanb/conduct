from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base
from models.types import JobStatus, Sensitivity


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    client_app_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("client_apps.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    sensitivity: Mapped[str] = mapped_column(
        String(20), nullable=False, default=Sensitivity.INTERNAL.value
    )
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=5, server_default="5")
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    model_requested: Mapped[str] = mapped_column(
        String(100), nullable=False, default="", server_default=""
    )
    model_used: Mapped[str] = mapped_column(
        String(100), nullable=False, default="", server_default=""
    )
    response: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=JobStatus.PENDING.value, index=True
    )
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    job_metadata: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default="{}"
    )
