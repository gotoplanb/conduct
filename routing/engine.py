"""Routing engine — pure functions from (job inputs, rule, defaults) to a decision.

No DB or HTTP. Routes/workers do the lookups, then call `decide()` with values.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.routing import RoutingRule
from models.types import Sensitivity, stricter
from providers.registry import is_cloud, provider_for_model


class SensitivityViolation(Exception):
    """A required model is disallowed by the job's effective sensitivity."""


@dataclass(frozen=True)
class RoutingDecision:
    model: str
    provider: str
    fallback_model: str | None
    fallback_provider: str | None
    effective_sensitivity: Sensitivity
    max_tokens: int
    reason: str


def _is_allowed(model: str, sensitivity: Sensitivity, allow_cloud_for_internal: bool) -> bool:
    if not is_cloud(model):
        return True
    if sensitivity == Sensitivity.CONFIDENTIAL:
        return False
    if sensitivity == Sensitivity.INTERNAL:
        return allow_cloud_for_internal
    return True  # public


def decide(
    *,
    sensitivity: Sensitivity,
    model_requested: str | None,
    allow_cloud_for_internal: bool,
    rule: RoutingRule | None,
    default_model: str,
    default_sensitive_model: str,
) -> RoutingDecision:
    # Rule sensitivity acts as a floor; clients can request stricter but not looser.
    rule_sensitivity = Sensitivity(rule.sensitivity) if rule else Sensitivity.INTERNAL
    effective = stricter(sensitivity, rule_sensitivity)

    max_tokens = rule.max_tokens if rule else 1000

    # 1. Explicit model override wins (subject to sensitivity).
    if model_requested:
        if not _is_allowed(model_requested, effective, allow_cloud_for_internal):
            raise SensitivityViolation(
                f"model {model_requested} disallowed for sensitivity={effective.value}"
            )
        return RoutingDecision(
            model=model_requested,
            provider=provider_for_model(model_requested),
            fallback_model=None,
            fallback_provider=None,
            effective_sensitivity=effective,
            max_tokens=max_tokens,
            reason="explicit-override",
        )

    # 2. Rule lookup, or fall back to env defaults.
    if rule:
        preferred = rule.preferred_model
        fallback: str | None = rule.fallback_model
        reason = f"rule:{rule.task_type}"
    else:
        preferred = (
            default_sensitive_model if effective == Sensitivity.CONFIDENTIAL else default_model
        )
        fallback = None
        reason = "default"

    preferred_ok = _is_allowed(preferred, effective, allow_cloud_for_internal)
    fallback_ok = fallback is not None and _is_allowed(
        fallback, effective, allow_cloud_for_internal
    )

    if not preferred_ok and not fallback_ok:
        raise SensitivityViolation(
            f"no model in rule for {rule.task_type if rule else '<default>'} "
            f"is allowed for sensitivity={effective.value}"
        )

    if not preferred_ok:
        # Fallback is eligible but preferred isn't — promote.
        assert fallback is not None
        preferred, fallback = fallback, None
        reason += "+sensitivity-promoted-fallback"
    elif fallback == preferred or not fallback_ok:
        # No useful fallback (same model, or fallback disallowed).
        fallback = None

    return RoutingDecision(
        model=preferred,
        provider=provider_for_model(preferred),
        fallback_model=fallback,
        fallback_provider=provider_for_model(fallback) if fallback else None,
        effective_sensitivity=effective,
        max_tokens=max_tokens,
        reason=reason,
    )
