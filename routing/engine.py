"""Routing engine — pure functions from (job inputs, rule, defaults) to a decision.

No DB or HTTP. Routes/workers do the lookups, then call `decide()` with values.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from models.routing import RoutingRule
from models.types import Sampling, Sensitivity, stricter
from providers.registry import is_cloud, provider_for_model


class SensitivityViolation(Exception):
    """A required model is disallowed by the job's effective sensitivity."""


# Profile → (temperature, derive_seed_from_input). Kept here, next to decide(),
# so the temperature/seed pair always moves together. `derive_seed` True means
# the executor seeds the model from a hash of the input (reproducible); False
# means leave the seed unset so each call varies.
_SAMPLING_PARAMS: dict[Sampling, tuple[float, bool]] = {
    Sampling.DETERMINISTIC: (0.0, True),
    Sampling.BALANCED: (0.7, False),
    Sampling.CREATIVE: (1.0, False),
}


def sampling_params(profile: Sampling) -> tuple[float, bool]:
    """Return (temperature, derive_seed_from_input) for a sampling profile."""
    return _SAMPLING_PARAMS[profile]


def rule_sampling(rule: RoutingRule | None) -> Sampling:
    """The rule's sampling profile, defaulting to BALANCED. Tolerates a
    missing rule and an in-memory rule whose column default hasn't been applied
    yet (SQLAlchemy fills server defaults only on flush, so `rule.sampling` can
    be None for a freshly-constructed row)."""
    if rule is not None and rule.sampling:
        return Sampling(rule.sampling)
    return Sampling.BALANCED


def derive_seed(prompt: str, system_prompt: str = "") -> int:
    """Deterministic seed from the input. Same prompt+context → same seed →
    (with temperature 0) reproducible output; different input → different seed.
    A positive 31-bit int, the safe range for an Ollama `options.seed`."""
    digest = hashlib.sha256(f"{system_prompt}\x00{prompt}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


@dataclass(frozen=True)
class RoutingDecision:
    model: str
    provider: str
    fallback_model: str | None
    fallback_provider: str | None
    effective_sensitivity: Sensitivity
    max_tokens: int
    reason: str
    temperature: float = 0.7
    # True → executor derives a seed from the job's input for reproducibility;
    # False → leave the seed unset so generations vary.
    deterministic_seed: bool = False


def _is_allowed(
    model: str,
    sensitivity: Sensitivity,
    allow_cloud_for_internal: bool,
    available_cloud_providers: frozenset[str],
) -> bool:
    if not is_cloud(model):
        return True
    if provider_for_model(model) not in available_cloud_providers:
        # Client has no credentials for this cloud provider (e.g. Bedrock
        # creds missing, or no Anthropic API key and no global fallback);
        # treat the model as unavailable for routing decisions.
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
    available_cloud_providers: frozenset[str],
    temperature: float,
    deterministic_seed: bool,
) -> RoutingDecision:
    """Build the decision when the caller specified a model directly. Raises
    SensitivityViolation if the requested model isn't allowed."""
    if not _is_allowed(
        model_requested, effective, allow_cloud_for_internal, available_cloud_providers
    ):
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
        temperature=temperature,
        deterministic_seed=deterministic_seed,
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
    available_cloud_providers: frozenset[str] = frozenset({"anthropic", "bedrock"}),
    sampling: Sampling | None = None,
) -> RoutingDecision:
    # Rule sensitivity acts as a floor; clients can request stricter but not looser.
    rule_sensitivity = Sensitivity(rule.sensitivity) if rule else Sensitivity.INTERNAL
    effective = stricter(sensitivity, rule_sensitivity)
    max_tokens = rule.max_tokens if rule else 1000

    # Sampling profile: per-request override wins, else the rule's, else the
    # global default. Resolves to a concrete (temperature, seed-policy) pair.
    profile = sampling or rule_sampling(rule)
    temperature, deterministic_seed = sampling_params(profile)

    if model_requested:
        return _explicit_override(
            model_requested, effective, allow_cloud_for_internal,
            max_tokens, available_cloud_providers, temperature, deterministic_seed,
        )

    preferred, fallback, reason = _rule_preferred_and_fallback(
        rule, effective, default_model, default_sensitive_model
    )
    preferred_ok = _is_allowed(
        preferred, effective, allow_cloud_for_internal, available_cloud_providers
    )
    fallback_ok = fallback is not None and _is_allowed(
        fallback, effective, allow_cloud_for_internal, available_cloud_providers
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
        temperature=temperature,
        deterministic_seed=deterministic_seed,
    )
