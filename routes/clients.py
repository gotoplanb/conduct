from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from auth import admin_only, generate_api_key, hash_api_key
from db.session import get_session
from models.client import ClientApp, ClientAppUsage
from secrets_box import SecretsKeyMissing, encrypt

router = APIRouter(prefix="/clients", tags=["clients"], dependencies=[Depends(admin_only)])

_CLIENT_NOT_FOUND = "client not found"


class ClientCreateIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    notes: str = ""
    rate_limit_per_minute: int | None = Field(default=None, ge=1)
    allow_cloud_for_internal: bool = False


class ClientCreateOut(BaseModel):
    id: UUID
    name: str
    api_key: str
    is_active: bool
    created_at: datetime


class ClientOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    is_active: bool
    rate_limit_per_minute: int | None
    allow_cloud_for_internal: bool
    notes: str
    created_at: datetime
    key_created_at: datetime
    anthropic_api_key_set_at: datetime | None = None


class RotateKeyOut(BaseModel):
    id: UUID
    name: str
    api_key: str
    key_created_at: datetime


class ClientPatchIn(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    is_active: bool | None = None
    notes: str | None = None
    rate_limit_per_minute: int | None = Field(default=None, ge=1)
    allow_cloud_for_internal: bool | None = None


class DayUsage(BaseModel):
    date: date
    job_count: int
    tokens_in: int
    tokens_out: int
    cost_usd: Decimal


class UsageOut(BaseModel):
    client_app_id: UUID
    period_days: int
    job_count: int
    tokens_in: int
    tokens_out: int
    cost_usd: Decimal
    by_day: list[DayUsage]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_client(
    body: ClientCreateIn, session: Annotated[AsyncSession, Depends(get_session)]
) -> ClientCreateOut:
    raw_key = generate_api_key()
    client = ClientApp(
        name=body.name,
        api_key_hash=hash_api_key(raw_key),
        notes=body.notes,
        rate_limit_per_minute=body.rate_limit_per_minute,
        allow_cloud_for_internal=body.allow_cloud_for_internal,
    )
    session.add(client)
    try:
        await session.commit()
    except IntegrityError as e:
        await session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "client name conflict") from e
    await session.refresh(client)
    return ClientCreateOut(
        id=client.id,
        name=client.name,
        api_key=raw_key,
        is_active=client.is_active,
        created_at=client.created_at,
    )


@router.get("", response_model=list[ClientOut])
async def list_clients(session: Annotated[AsyncSession, Depends(get_session)]) -> list[ClientApp]:
    result = await session.scalars(select(ClientApp).order_by(ClientApp.created_at))
    return list(result.all())


@router.patch("/{client_id}", response_model=ClientOut)
async def patch_client(
    client_id: UUID,
    body: ClientPatchIn,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ClientApp:
    client = await session.get(ClientApp, client_id)
    if client is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, _CLIENT_NOT_FOUND)
    updates = body.model_dump(exclude_unset=True)
    for k, v in updates.items():
        setattr(client, k, v)
    await session.commit()
    await session.refresh(client)
    return client


@router.post("/{client_id}/rotate-key")
async def rotate_key(
    client_id: UUID, session: Annotated[AsyncSession, Depends(get_session)]
) -> RotateKeyOut:
    """Mint a fresh API key for an existing client. The old key stops working
    immediately. The raw key is returned once here and never again."""
    client = await session.get(ClientApp, client_id)
    if client is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, _CLIENT_NOT_FOUND)
    raw_key = generate_api_key()
    client.api_key_hash = hash_api_key(raw_key)
    client.key_created_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(client)
    return RotateKeyOut(
        id=client.id,
        name=client.name,
        api_key=raw_key,
        key_created_at=client.key_created_at,
    )


class AnthropicKeyIn(BaseModel):
    api_key: str = Field(min_length=1)


class AnthropicKeyOut(BaseModel):
    id: UUID
    name: str
    anthropic_api_key_set_at: datetime | None


@router.put("/{client_id}/anthropic-key")
async def set_anthropic_key(
    client_id: UUID,
    body: AnthropicKeyIn,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AnthropicKeyOut:
    """Store a per-client Anthropic API key, encrypted with CONDUCT_SECRETS_KEY.
    The plaintext is never persisted or echoed back — only the set timestamp."""
    client = await session.get(ClientApp, client_id)
    if client is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, _CLIENT_NOT_FOUND)
    try:
        client.anthropic_api_key_encrypted = encrypt(body.api_key.strip())
    except SecretsKeyMissing as e:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "CONDUCT_SECRETS_KEY is not configured on the server",
        ) from e
    client.anthropic_api_key_set_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(client)
    return AnthropicKeyOut(
        id=client.id,
        name=client.name,
        anthropic_api_key_set_at=client.anthropic_api_key_set_at,
    )


@router.delete("/{client_id}/anthropic-key")
async def clear_anthropic_key(
    client_id: UUID, session: Annotated[AsyncSession, Depends(get_session)]
) -> AnthropicKeyOut:
    """Forget this client's Anthropic key — they fall back to local-only routing."""
    client = await session.get(ClientApp, client_id)
    if client is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, _CLIENT_NOT_FOUND)
    client.anthropic_api_key_encrypted = None
    client.anthropic_api_key_set_at = None
    await session.commit()
    await session.refresh(client)
    return AnthropicKeyOut(
        id=client.id,
        name=client.name,
        anthropic_api_key_set_at=None,
    )


@router.get("/{client_id}/usage")
async def client_usage(
    client_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    days: Annotated[int, Query(ge=1, le=365)] = 30,
) -> UsageOut:
    client = await session.get(ClientApp, client_id)
    if client is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, _CLIENT_NOT_FOUND)

    today = datetime.now(UTC).date()
    since = today - timedelta(days=days - 1)

    rows = (
        await session.scalars(
            select(ClientAppUsage)
            .where(
                ClientAppUsage.client_app_id == client_id,
                ClientAppUsage.date >= since,
            )
            .order_by(ClientAppUsage.date)
        )
    ).all()

    totals = await session.execute(
        select(
            func.coalesce(func.sum(ClientAppUsage.job_count), 0),
            func.coalesce(func.sum(ClientAppUsage.tokens_in), 0),
            func.coalesce(func.sum(ClientAppUsage.tokens_out), 0),
            func.coalesce(func.sum(ClientAppUsage.cost_usd), 0),
        ).where(
            ClientAppUsage.client_app_id == client_id,
            ClientAppUsage.date >= since,
        )
    )
    job_count, tokens_in, tokens_out, cost_usd = totals.one()

    return UsageOut(
        client_app_id=client_id,
        period_days=days,
        job_count=job_count,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=Decimal(cost_usd),
        by_day=[
            DayUsage(
                date=r.date,
                job_count=r.job_count,
                tokens_in=r.tokens_in,
                tokens_out=r.tokens_out,
                cost_usd=r.cost_usd,
            )
            for r in rows
        ],
    )
