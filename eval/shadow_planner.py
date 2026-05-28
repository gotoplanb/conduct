"""Pure shadow-target planner.

Given a parent job's context, decides which shadow models (if any) to enqueue.
No DB or HTTP — caller pre-fetches the rule + today's cost-per-model and the
planner just applies the gates: sensitivity, sample rate, daily cost cap.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal

from models.routing import RoutingRule
from models.types import Sensitivity
from providers.registry import is_cloud, provider_for_model


@dataclass(frozen=True)
class ShadowPlan:
    """A single shadow-job target that passed all gates."""

    model: str
    provider: str


def _cloud_target_blocked(
    *,
    model: str,
    spec: dict,
    sensitivity: Sensitivity,
    allow_cloud_for_internal: bool,
    today_cost_by_model: dict[str, Decimal],
) -> bool:
    """Sensitivity + daily-cost gates for a cloud shadow target. Local models
    skip this entirely."""
    if sensitivity == Sensitivity.CONFIDENTIAL:
        return True
    if sensitivity == Sensitivity.INTERNAL and not allow_cloud_for_internal:
        return True
    cap = spec.get("daily_cost_cap_usd")
    if cap is not None:
        spent = Decimal(str(today_cost_by_model.get(model, 0)))
        if spent >= Decimal(str(cap)):
            return True
    return False


def _sampled_in(spec: dict, rng: random.Random) -> bool:
    rate = float(spec.get("rate", 0))
    return rate > 0 and rng.random() < rate


def plan_shadows(
    *,
    rule: RoutingRule | None,
    sensitivity: Sensitivity,
    allow_cloud_for_internal: bool,
    primary_model: str,
    today_cost_by_model: dict[str, Decimal],
    rng: random.Random | None = None,
    force_all: bool = False,
) -> list[ShadowPlan]:
    """Return the shadow targets to enqueue for this parent job.

    Gates, in order: rule present, target != primary, sensitivity (cloud
    only), daily cost cap (cloud only), per-target sample rate. When
    `force_all` is true the per-target rate is bypassed — useful when the
    caller explicitly asked to fan out for this specific request.
    """
    if rule is None or not rule.eval_shadow_models:
        return []

    r = rng or random.Random()
    plans: list[ShadowPlan] = []
    for spec in rule.eval_shadow_models:
        model = spec.get("model")
        if not model or model == primary_model:
            continue
        if is_cloud(model) and _cloud_target_blocked(
            model=model,
            spec=spec,
            sensitivity=sensitivity,
            allow_cloud_for_internal=allow_cloud_for_internal,
            today_cost_by_model=today_cost_by_model,
        ):
            continue
        if not force_all and not _sampled_in(spec, r):
            continue
        plans.append(ShadowPlan(model=model, provider=provider_for_model(model)))
    return plans
