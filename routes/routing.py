from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import admin_only
from db.session import get_session
from models.routing import RoutingRule
from models.types import Sensitivity

router = APIRouter(prefix="/routing", tags=["routing"], dependencies=[Depends(admin_only)])


class RoutingRuleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_type: str
    preferred_model: str
    fallback_model: str
    sensitivity: Sensitivity
    max_tokens: int
    notes: str
    updated_at: datetime


class RoutingRuleIn(BaseModel):
    preferred_model: str = Field(min_length=1, max_length=100)
    fallback_model: str = Field(min_length=1, max_length=100)
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    max_tokens: int = Field(default=1000, ge=1, le=200_000)
    notes: str = ""


class RoutingListOut(BaseModel):
    rules: list[RoutingRuleOut]


@router.get("", response_model=RoutingListOut)
async def list_routing(session: AsyncSession = Depends(get_session)) -> RoutingListOut:
    rows = (await session.scalars(select(RoutingRule).order_by(RoutingRule.task_type))).all()
    return RoutingListOut(rules=[RoutingRuleOut.model_validate(r) for r in rows])


@router.put("/{task_type}", response_model=RoutingRuleOut)
async def upsert_routing(
    task_type: str,
    body: RoutingRuleIn,
    session: AsyncSession = Depends(get_session),
) -> RoutingRule:
    if not task_type or len(task_type) > 100:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "task_type must be 1-100 chars")
    rule = await session.get(RoutingRule, task_type)
    if rule is None:
        rule = RoutingRule(
            task_type=task_type,
            preferred_model=body.preferred_model,
            fallback_model=body.fallback_model,
            sensitivity=body.sensitivity.value,
            max_tokens=body.max_tokens,
            notes=body.notes,
        )
        session.add(rule)
    else:
        rule.preferred_model = body.preferred_model
        rule.fallback_model = body.fallback_model
        rule.sensitivity = body.sensitivity.value
        rule.max_tokens = body.max_tokens
        rule.notes = body.notes
    await session.commit()
    await session.refresh(rule)
    return rule
