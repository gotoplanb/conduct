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


def plan_shadows(
    *,
    rule: RoutingRule | None,
    sensitivity: Sensitivity,
    allow_cloud_for_internal: bool,
    primary_model: str,
    today_cost_by_model: dict[str, Decimal],
    rng: random.Random | None = None,
) -> list[ShadowPlan]:
    """Return the shadow targets to enqueue for this parent job.

    Gates, in order:
      1. Rule has `eval_shadow_models` configured (else: no shadows).
      2. Target model != primary model (no point shadowing yourself).
      3. Sensitivity: confidential never shadows cloud; internal only when
         the client opted in via `allow_cloud_for_internal`.
      4. Daily cost cap (per-model). Local models have no cap.
      5. Random sample using the per-target rate.
    """
    if rule is None or not rule.eval_shadow_models:
        return []

    r = rng or random.Random()
    plans: list[ShadowPlan] = []
    for spec in rule.eval_shadow_models:
        model = spec.get("model")
        if not model or model == primary_model:
            continue

        if is_cloud(model):
            if sensitivity == Sensitivity.CONFIDENTIAL:
                continue
            if sensitivity == Sensitivity.INTERNAL and not allow_cloud_for_internal:
                continue
            cap = spec.get("daily_cost_cap_usd")
            if cap is not None:
                spent = Decimal(str(today_cost_by_model.get(model, 0)))
                if spent >= Decimal(str(cap)):
                    continue

        rate = float(spec.get("rate", 0))
        if rate <= 0 or r.random() >= rate:
            continue

        plans.append(ShadowPlan(model=model, provider=provider_for_model(model)))
    return plans
