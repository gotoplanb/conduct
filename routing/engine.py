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


def _is_allowed(
    model: str,
    sensitivity: Sensitivity,
    allow_cloud_for_internal: bool,
    cloud_available: bool = True,
) -> bool:
    if not is_cloud(model):
        return True
    if not cloud_available:
        # Client has no Anthropic key (and no global fallback registered);
        # treat cloud as unavailable for routing decisions.
        return False
    if sensitivity == Sensitivity.CONFIDENTIAL:
        return False
    if sensitivity == Sensitivity.INTERNAL:
        return allow_cloud_for_internal
    return True  # public


def _explicit_override(
    model_requested: str,
    effective: Sensitivity,
    allow_cloud_for_internal: bool,
    max_tokens: int,
    cloud_available: bool = True,
) -> RoutingDecision:
    """Build the decision when the caller specified a model directly. Raises
    SensitivityViolation if the requested model isn't allowed."""
    if not _is_allowed(model_requested, effective, allow_cloud_for_internal, cloud_available):
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


def _rule_preferred_and_fallback(
    rule: RoutingRule | None,
    effective: Sensitivity,
    default_model: str,
    default_sensitive_model: str,
) -> tuple[str, str | None, str]:
    """Pick (preferred, fallback, reason) from either the rule or env defaults."""
    if rule:
        return rule.preferred_model, rule.fallback_model, f"rule:{rule.task_type}"
    preferred = (
        default_sensitive_model if effective == Sensitivity.CONFIDENTIAL else default_model
    )
    return preferred, None, "default"


def decide(
    *,
    sensitivity: Sensitivity,
    model_requested: str | None,
    allow_cloud_for_internal: bool,
    rule: RoutingRule | None,
    default_model: str,
    default_sensitive_model: str,
    cloud_available: bool = True,
) -> RoutingDecision:
    # Rule sensitivity acts as a floor; clients can request stricter but not looser.
    rule_sensitivity = Sensitivity(rule.sensitivity) if rule else Sensitivity.INTERNAL
    effective = stricter(sensitivity, rule_sensitivity)
    max_tokens = rule.max_tokens if rule else 1000

    if model_requested:
        return _explicit_override(
            model_requested, effective, allow_cloud_for_internal, max_tokens, cloud_available
        )

    preferred, fallback, reason = _rule_preferred_and_fallback(
        rule, effective, default_model, default_sensitive_model
    )
    preferred_ok = _is_allowed(preferred, effective, allow_cloud_for_internal, cloud_available)
    fallback_ok = fallback is not None and _is_allowed(
        fallback, effective, allow_cloud_for_internal, cloud_available
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
