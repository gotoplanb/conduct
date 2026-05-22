"""Seeder coverage for routing rules — focused on the eval_shadow_models
pass-through, since that field is easy to silently drop. Uses unique
task_types so it doesn't collide with rules already seeded in the test DB."""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import select

from models.routing import RoutingRule
from scripts.seed import _seed_routing


@pytest.fixture
def task_type() -> str:
    return f"seed-test-{uuid4().hex[:8]}"


async def test_seed_routing_carries_shadow_models(db_session, task_type) -> None:
    specs = [
        {
            "task_type": task_type,
            "preferred_model": "llama3.3:70b",
            "fallback_model": "claude-sonnet-4-5",
            "sensitivity": "internal",
            "eval_shadow_models": [{"model": "llama3.2:3b", "rate": 1.0}],
        }
    ]
    created, skipped = await _seed_routing(db_session, specs)
    await db_session.commit()

    assert created == [task_type]
    assert skipped == []
    rule = await db_session.scalar(
        select(RoutingRule).where(RoutingRule.task_type == task_type)
    )
    assert rule.eval_shadow_models == [{"model": "llama3.2:3b", "rate": 1.0}]


async def test_seed_routing_defaults_shadow_models_to_empty(db_session, task_type) -> None:
    specs = [
        {
            "task_type": task_type,
            "preferred_model": "claude-sonnet-4-5",
            "fallback_model": "claude-opus-4-5",
            "sensitivity": "public",
        }
    ]
    await _seed_routing(db_session, specs)
    await db_session.commit()

    rule = await db_session.scalar(
        select(RoutingRule).where(RoutingRule.task_type == task_type)
    )
    assert rule.eval_shadow_models == []


async def test_seed_routing_skips_existing(db_session, task_type) -> None:
    spec = {
        "task_type": task_type,
        "preferred_model": "llama3.3:70b",
        "fallback_model": "claude-sonnet-4-5",
        "sensitivity": "internal",
    }
    created, _ = await _seed_routing(db_session, [spec])
    await db_session.commit()
    assert created == [task_type]

    # Second pass with a different shadow config must NOT overwrite.
    spec_changed = {**spec, "eval_shadow_models": [{"model": "x", "rate": 0.5}]}
    created, skipped = await _seed_routing(db_session, [spec_changed])
    await db_session.commit()

    assert created == []
    assert skipped == [task_type]
    rule = await db_session.scalar(
        select(RoutingRule).where(RoutingRule.task_type == task_type)
    )
    assert rule.eval_shadow_models == []
