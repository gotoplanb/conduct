"""Tests for the shadow-job planner (pure function, no DB)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest

from eval.shadow_planner import plan_shadows
from models.types import Sensitivity


@dataclass
class FakeRule:
    """Stand-in for RoutingRule that only carries the fields the planner uses."""

    eval_shadow_models: list[dict[str, Any]]


def _always_zero() -> dict[str, Decimal]:
    return {}


# --- baseline behavior ------------------------------------------------------


def test_no_rule_returns_empty() -> None:
    assert (
        plan_shadows(
            rule=None,
            sensitivity=Sensitivity.INTERNAL,
            allow_cloud_for_internal=True,
            primary_model="llama3.3:70b",
            today_cost_by_model=_always_zero(),
        )
        == []
    )


def test_empty_eval_shadow_models_returns_empty() -> None:
    rule = FakeRule(eval_shadow_models=[])
    assert (
        plan_shadows(
            rule=rule,  # type: ignore[arg-type]
            sensitivity=Sensitivity.INTERNAL,
            allow_cloud_for_internal=True,
            primary_model="llama3.3:70b",
            today_cost_by_model=_always_zero(),
        )
        == []
    )


def test_picks_target_when_sample_rate_hits() -> None:
    rule = FakeRule(eval_shadow_models=[{"model": "qwen2.5:72b", "rate": 1.0}])
    plans = plan_shadows(
        rule=rule,  # type: ignore[arg-type]
        sensitivity=Sensitivity.INTERNAL,
        allow_cloud_for_internal=False,
        primary_model="llama3.3:70b",
        today_cost_by_model=_always_zero(),
        rng=random.Random(0),
    )
    assert len(plans) == 1
    assert plans[0].model == "qwen2.5:72b"
    assert plans[0].provider == "ollama"


def test_skips_target_when_sample_rate_misses() -> None:
    # rate 0 → never picks
    rule = FakeRule(eval_shadow_models=[{"model": "qwen2.5:72b", "rate": 0.0}])
    plans = plan_shadows(
        rule=rule,  # type: ignore[arg-type]
        sensitivity=Sensitivity.INTERNAL,
        allow_cloud_for_internal=False,
        primary_model="llama3.3:70b",
        today_cost_by_model=_always_zero(),
        rng=random.Random(0),
    )
    assert plans == []


def test_skips_target_matching_primary() -> None:
    # No point shadowing yourself.
    rule = FakeRule(eval_shadow_models=[{"model": "llama3.3:70b", "rate": 1.0}])
    plans = plan_shadows(
        rule=rule,  # type: ignore[arg-type]
        sensitivity=Sensitivity.INTERNAL,
        allow_cloud_for_internal=False,
        primary_model="llama3.3:70b",
        today_cost_by_model=_always_zero(),
        rng=random.Random(0),
    )
    assert plans == []


# --- sensitivity gates ------------------------------------------------------


def test_confidential_blocks_cloud_shadow() -> None:
    rule = FakeRule(eval_shadow_models=[{"model": "claude-haiku-4-5", "rate": 1.0}])
    plans = plan_shadows(
        rule=rule,  # type: ignore[arg-type]
        sensitivity=Sensitivity.CONFIDENTIAL,
        allow_cloud_for_internal=True,  # ignored for confidential
        primary_model="llama3.3:70b",
        today_cost_by_model=_always_zero(),
        rng=random.Random(0),
    )
    assert plans == []


def test_confidential_allows_local_shadow() -> None:
    rule = FakeRule(eval_shadow_models=[{"model": "qwen2.5:72b", "rate": 1.0}])
    plans = plan_shadows(
        rule=rule,  # type: ignore[arg-type]
        sensitivity=Sensitivity.CONFIDENTIAL,
        allow_cloud_for_internal=False,
        primary_model="llama3.3:70b",
        today_cost_by_model=_always_zero(),
        rng=random.Random(0),
    )
    assert len(plans) == 1
    assert plans[0].model == "qwen2.5:72b"


def test_internal_respects_allow_cloud_flag() -> None:
    rule = FakeRule(eval_shadow_models=[{"model": "claude-haiku-4-5", "rate": 1.0}])
    plans_blocked = plan_shadows(
        rule=rule,  # type: ignore[arg-type]
        sensitivity=Sensitivity.INTERNAL,
        allow_cloud_for_internal=False,
        primary_model="llama3.3:70b",
        today_cost_by_model=_always_zero(),
        rng=random.Random(0),
    )
    assert plans_blocked == []

    plans_allowed = plan_shadows(
        rule=rule,  # type: ignore[arg-type]
        sensitivity=Sensitivity.INTERNAL,
        allow_cloud_for_internal=True,
        primary_model="llama3.3:70b",
        today_cost_by_model=_always_zero(),
        rng=random.Random(0),
    )
    assert len(plans_allowed) == 1


def test_public_allows_cloud_regardless_of_flag() -> None:
    rule = FakeRule(eval_shadow_models=[{"model": "claude-haiku-4-5", "rate": 1.0}])
    plans = plan_shadows(
        rule=rule,  # type: ignore[arg-type]
        sensitivity=Sensitivity.PUBLIC,
        allow_cloud_for_internal=False,  # irrelevant for public
        primary_model="llama3.3:70b",
        today_cost_by_model=_always_zero(),
        rng=random.Random(0),
    )
    assert len(plans) == 1


# --- cost cap ---------------------------------------------------------------


def test_cost_cap_blocks_cloud_when_exceeded() -> None:
    rule = FakeRule(
        eval_shadow_models=[
            {"model": "claude-haiku-4-5", "rate": 1.0, "daily_cost_cap_usd": 0.50}
        ]
    )
    plans = plan_shadows(
        rule=rule,  # type: ignore[arg-type]
        sensitivity=Sensitivity.PUBLIC,
        allow_cloud_for_internal=True,
        primary_model="llama3.3:70b",
        today_cost_by_model={"claude-haiku-4-5": Decimal("0.51")},
        rng=random.Random(0),
    )
    assert plans == []


def test_cost_cap_allows_when_under() -> None:
    rule = FakeRule(
        eval_shadow_models=[
            {"model": "claude-haiku-4-5", "rate": 1.0, "daily_cost_cap_usd": 1.00}
        ]
    )
    plans = plan_shadows(
        rule=rule,  # type: ignore[arg-type]
        sensitivity=Sensitivity.PUBLIC,
        allow_cloud_for_internal=True,
        primary_model="llama3.3:70b",
        today_cost_by_model={"claude-haiku-4-5": Decimal("0.40")},
        rng=random.Random(0),
    )
    assert len(plans) == 1


def test_cost_cap_ignored_for_local_models() -> None:
    # Local models have no marginal cost; cap should be a no-op.
    rule = FakeRule(
        eval_shadow_models=[{"model": "qwen2.5:72b", "rate": 1.0, "daily_cost_cap_usd": 0.01}]
    )
    plans = plan_shadows(
        rule=rule,  # type: ignore[arg-type]
        sensitivity=Sensitivity.INTERNAL,
        allow_cloud_for_internal=False,
        primary_model="llama3.3:70b",
        today_cost_by_model={"qwen2.5:72b": Decimal("999.99")},
        rng=random.Random(0),
    )
    assert len(plans) == 1


# --- multiple targets -------------------------------------------------------


def test_multiple_targets_evaluated_independently() -> None:
    # One target eligible, one blocked by sensitivity. RNG seeded so rate=1 hits.
    rule = FakeRule(
        eval_shadow_models=[
            {"model": "qwen2.5:72b", "rate": 1.0},  # local — passes
            {"model": "claude-haiku-4-5", "rate": 1.0},  # cloud — confidential blocks
        ]
    )
    plans = plan_shadows(
        rule=rule,  # type: ignore[arg-type]
        sensitivity=Sensitivity.CONFIDENTIAL,
        allow_cloud_for_internal=True,
        primary_model="llama3.3:70b",
        today_cost_by_model=_always_zero(),
        rng=random.Random(0),
    )
    assert [p.model for p in plans] == ["qwen2.5:72b"]


def test_zero_or_negative_rate_skipped() -> None:
    rule = FakeRule(
        eval_shadow_models=[
            {"model": "qwen2.5:72b", "rate": 0.0},
            {"model": "llama3.2:3b", "rate": -0.5},
        ]
    )
    plans = plan_shadows(
        rule=rule,  # type: ignore[arg-type]
        sensitivity=Sensitivity.PUBLIC,
        allow_cloud_for_internal=True,
        primary_model="llama3.3:70b",
        today_cost_by_model=_always_zero(),
        rng=random.Random(0),
    )
    assert plans == []


def test_missing_model_field_skipped() -> None:
    rule = FakeRule(eval_shadow_models=[{"rate": 1.0}, {"model": "", "rate": 1.0}])
    plans = plan_shadows(
        rule=rule,  # type: ignore[arg-type]
        sensitivity=Sensitivity.PUBLIC,
        allow_cloud_for_internal=True,
        primary_model="llama3.3:70b",
        today_cost_by_model=_always_zero(),
        rng=random.Random(0),
    )
    assert plans == []


# --- sample-rate distribution sanity check ----------------------------------


@pytest.mark.parametrize("seed", [0, 1, 42, 100])
def test_sample_rate_distribution_with_fixed_seed(seed: int) -> None:
    """At rate=0.5 over many trials we should see roughly half pass."""
    rule = FakeRule(eval_shadow_models=[{"model": "qwen2.5:72b", "rate": 0.5}])
    rng = random.Random(seed)
    hits = 0
    n = 1000
    for _ in range(n):
        plans = plan_shadows(
            rule=rule,  # type: ignore[arg-type]
            sensitivity=Sensitivity.PUBLIC,
            allow_cloud_for_internal=True,
            primary_model="llama3.3:70b",
            today_cost_by_model=_always_zero(),
            rng=rng,
        )
        if plans:
            hits += 1
    # Generous tolerance — we just want to confirm the sampling is roughly fair.
    assert 400 <= hits <= 600


def test_force_all_bypasses_rate() -> None:
    # rate=0 would never sample in normally; force_all should bypass that.
    rule = FakeRule(eval_shadow_models=[{"model": "qwen2.5:14b", "rate": 0.0}])
    plans = plan_shadows(
        rule=rule,
        sensitivity=Sensitivity.INTERNAL,
        allow_cloud_for_internal=True,
        primary_model="llama3.3:70b",
        today_cost_by_model=_always_zero(),
        rng=random.Random(0),
        force_all=True,
    )
    assert [p.model for p in plans] == ["qwen2.5:14b"]


def test_force_all_still_skips_primary_and_blocked_cloud() -> None:
    # force_all bypasses RATE; it must not bypass sensitivity or primary-self.
    rule = FakeRule(
        eval_shadow_models=[
            {"model": "llama3.3:70b", "rate": 1.0},  # same as primary → skip
            {"model": "claude-haiku-4-5", "rate": 1.0},  # cloud + confidential → blocked
            {"model": "qwen2.5:14b", "rate": 0.0},
        ]
    )
    plans = plan_shadows(
        rule=rule,
        sensitivity=Sensitivity.CONFIDENTIAL,
        allow_cloud_for_internal=False,
        primary_model="llama3.3:70b",
        today_cost_by_model=_always_zero(),
        rng=random.Random(0),
        force_all=True,
    )
    assert [p.model for p in plans] == ["qwen2.5:14b"]
