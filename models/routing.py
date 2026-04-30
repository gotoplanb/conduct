from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base
from models.types import Sensitivity


class RoutingRule(Base):
    __tablename__ = "routing_rules"

    task_type: Mapped[str] = mapped_column(String(100), primary_key=True)
    preferred_model: Mapped[str] = mapped_column(String(100), nullable=False)
    fallback_model: Mapped[str] = mapped_column(String(100), nullable=False)
    sensitivity: Mapped[str] = mapped_column(
        String(20), nullable=False, default=Sensitivity.INTERNAL.value
    )
    max_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1000, server_default="1000"
    )
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
