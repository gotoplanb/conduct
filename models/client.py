from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class ClientApp(Base):
    __tablename__ = "client_apps"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    api_key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    rate_limit_per_minute: Mapped[int | None] = mapped_column(Integer, nullable=True)
    allow_cloud_for_internal: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class ClientAppUsage(Base):
    __tablename__ = "client_app_usage"
    __table_args__ = (UniqueConstraint("client_app_id", "date", name="uq_usage_client_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_app_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("client_apps.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    tokens_in: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default="0"
    )
    tokens_out: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default="0"
    )
    job_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(10, 4), nullable=False, default=Decimal("0"), server_default="0"
    )
