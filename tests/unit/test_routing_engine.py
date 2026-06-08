from __future__ import annotations

import pytest

from models.routing import RoutingRule
from models.types import Sampling, Sensitivity
from routing.engine import (
    SensitivityViolation,
    decide,
    derive_seed,
    rule_sampling,
    sampling_params,
)


def _rule(
    task_type: str = "x",
    preferred: str = "llama3.3:70b",
    fallback: str = "claude-sonnet-4-5",
    sensitivity: Sensitivity = Sensitivity.INTERNAL,
    max_tokens: int = 1000,
    sampling: str | None = None,
) -> RoutingRule:
    r = RoutingRule(
        task_type=task_type,
        preferred_model=preferred,
        fallback_model=fallback,
        sensitivity=sensitivity.value,
        max_tokens=max_tokens,
        sampling=sampling,
    )
    return r


def test_explicit_override_wins() -> None:
    d = decide(
        sensitivity=Sensitivity.PUBLIC,
        model_requested="claude-haiku-4-5",
        allow_cloud_for_internal=False,
        rule=_rule(sensitivity=Sensitivity.PUBLIC),
        default_model="llama3.3:70b",
        default_sensitive_model="llama3.3:70b",
    )
    assert d.model == "claude-haiku-4-5"
    assert d.provider == "anthropic"
    assert d.fallback_model is None
    assert d.reason == "explicit-override"


def test_confidential_blocks_explicit_cloud_override() -> None:
    with pytest.raises(SensitivityViolation):
        decide(
            sensitivity=Sensitivity.CONFIDENTIAL,
            model_requested="claude-sonnet-4-5",
            allow_cloud_for_internal=True,
            rule=_rule(sensitivity=Sensitivity.INTERNAL),
            default_model="llama3.3:70b",
            default_sensitive_model="llama3.3:70b",
        )


def test_internal_blocks_cloud_unless_client_opts_in() -> None:
    with pytest.raises(SensitivityViolation):
        decide(
            sensitivity=Sensitivity.INTERNAL,
            model_requested="claude-sonnet-4-5",
            allow_cloud_for_internal=False,
            rule=None,
            default_model="llama3.3:70b",
            default_sensitive_model="llama3.3:70b",
        )

    d = decide(
        sensitivity=Sensitivity.INTERNAL,
        model_requested="claude-sonnet-4-5",
        allow_cloud_for_internal=True,
        rule=None,
        default_model="llama3.3:70b",
        default_sensitive_model="llama3.3:70b",
    )
    assert d.model == "claude-sonnet-4-5"


def test_rule_used_when_no_override() -> None:
    d = decide(
        sensitivity=Sensitivity.INTERNAL,
        model_requested=None,
        allow_cloud_for_internal=False,
        rule=_rule(preferred="qwen2.5:72b", fallback="claude-sonnet-4-5"),
        default_model="llama3.3:70b",
        default_sensitive_model="llama3.3:70b",
    )
    assert d.model == "qwen2.5:72b"
    assert d.provider == "ollama"
    # fallback dropped because client doesn't allow cloud for internal
    assert d.fallback_model is None


def test_internal_with_opt_in_keeps_cloud_fallback() -> None:
    d = decide(
        sensitivity=Sensitivity.INTERNAL,
        model_requested=None,
        allow_cloud_for_internal=True,
        rule=_rule(preferred="llama3.3:70b", fallback="claude-sonnet-4-5"),
        default_model="llama3.3:70b",
        default_sensitive_model="llama3.3:70b",
    )
    assert d.model == "llama3.3:70b"
    assert d.fallback_model == "claude-sonnet-4-5"
    assert d.fallback_provider == "anthropic"


def test_confidential_drops_cloud_fallback_keeps_local() -> None:
    d = decide(
        sensitivity=Sensitivity.CONFIDENTIAL,
        model_requested=None,
        allow_cloud_for_internal=True,  # ignored — confidential is hard-blocked
        rule=_rule(preferred="llama3.3:70b", fallback="claude-sonnet-4-5"),
        default_model="llama3.3:70b",
        default_sensitive_model="llama3.3:70b",
    )
    assert d.model == "llama3.3:70b"
    assert d.fallback_model is None  # cloud fallback dropped


def test_rule_sensitivity_acts_as_floor() -> None:
    # Client requests public but rule says confidential — effective is confidential.
    d = decide(
        sensitivity=Sensitivity.PUBLIC,
        model_requested=None,
        allow_cloud_for_internal=True,
        rule=_rule(
            preferred="llama3.3:70b",
            fallback="claude-sonnet-4-5",
            sensitivity=Sensitivity.CONFIDENTIAL,
        ),
        default_model="llama3.3:70b",
        default_sensitive_model="llama3.3:70b",
    )
    assert d.effective_sensitivity == Sensitivity.CONFIDENTIAL
    assert d.fallback_model is None  # confidential strips cloud fallback


def test_no_rule_uses_default() -> None:
    d = decide(
        sensitivity=Sensitivity.PUBLIC,
        model_requested=None,
        allow_cloud_for_internal=False,
        rule=None,
        default_model="llama3.3:70b",
        default_sensitive_model="llama3.3:70b",
    )
    assert d.model == "llama3.3:70b"
    assert d.reason == "default"


def test_no_rule_confidential_uses_sensitive_default() -> None:
    d = decide(
        sensitivity=Sensitivity.CONFIDENTIAL,
        model_requested=None,
        allow_cloud_for_internal=False,
        rule=None,
        default_model="claude-sonnet-4-5",  # cloud — would be blocked
        default_sensitive_model="llama3.3:70b",  # local default for sensitive
    )
    assert d.model == "llama3.3:70b"


def test_all_models_blocked_raises() -> None:
    with pytest.raises(SensitivityViolation):
        decide(
            sensitivity=Sensitivity.CONFIDENTIAL,
            model_requested=None,
            allow_cloud_for_internal=False,
            rule=_rule(
                preferred="claude-sonnet-4-5",
                fallback="claude-opus-4-5",
                sensitivity=Sensitivity.PUBLIC,
            ),
            default_model="llama3.3:70b",
            default_sensitive_model="llama3.3:70b",
        )


def test_max_tokens_propagated_from_rule() -> None:
    d = decide(
        sensitivity=Sensitivity.PUBLIC,
        model_requested=None,
        allow_cloud_for_internal=False,
        rule=_rule(max_tokens=4000),
        default_model="llama3.3:70b",
        default_sensitive_model="llama3.3:70b",
    )
    assert d.max_tokens == 4000


def test_max_tokens_default_when_no_rule() -> None:
    d = decide(
        sensitivity=Sensitivity.PUBLIC,
        model_requested=None,
        allow_cloud_for_internal=False,
        rule=None,
        default_model="llama3.3:70b",
        default_sensitive_model="llama3.3:70b",
    )
    assert d.max_tokens == 1000


# --- sampling profile ---


def _decide_with_sampling(rule, sampling=None):
    return decide(
        sensitivity=Sensitivity.PUBLIC,
        model_requested=None,
        allow_cloud_for_internal=False,
        rule=rule,
        default_model="llama3.3:70b",
        default_sensitive_model="llama3.3:70b",
        sampling=sampling,
    )


def test_sampling_params_table() -> None:
    assert sampling_params(Sampling.DETERMINISTIC) == (0.0, True)
    assert sampling_params(Sampling.BALANCED) == (0.7, False)
    assert sampling_params(Sampling.CREATIVE) == (1.0, False)


def test_deterministic_profile_from_rule_sets_temp0_and_seed_flag() -> None:
    d = _decide_with_sampling(_rule(sampling="deterministic"))
    assert d.temperature == 0.0
    assert d.deterministic_seed is True


def test_creative_profile_from_rule() -> None:
    d = _decide_with_sampling(_rule(sampling="creative"))
    assert d.temperature == 1.0
    assert d.deterministic_seed is False


def test_per_request_sampling_overrides_rule() -> None:
    # Rule says deterministic, request asks for creative → creative wins.
    d = _decide_with_sampling(_rule(sampling="deterministic"), sampling=Sampling.CREATIVE)
    assert d.temperature == 1.0
    assert d.deterministic_seed is False


def test_sampling_defaults_to_balanced_when_unset() -> None:
    # In-memory rule with sampling=None (column default not yet applied) and no
    # request override → balanced, not a crash.
    d = _decide_with_sampling(_rule(sampling=None))
    assert d.temperature == 0.7
    assert d.deterministic_seed is False


def test_explicit_model_override_still_carries_sampling() -> None:
    d = decide(
        sensitivity=Sensitivity.PUBLIC,
        model_requested="claude-haiku-4-5",
        allow_cloud_for_internal=False,
        rule=_rule(sensitivity=Sensitivity.PUBLIC, sampling="deterministic"),
        default_model="llama3.3:70b",
        default_sensitive_model="llama3.3:70b",
    )
    assert d.reason == "explicit-override"
    assert d.temperature == 0.0
    assert d.deterministic_seed is True


def test_rule_sampling_helper_tolerates_none_and_missing() -> None:
    assert rule_sampling(None) is Sampling.BALANCED
    assert rule_sampling(_rule(sampling=None)) is Sampling.BALANCED
    assert rule_sampling(_rule(sampling="creative")) is Sampling.CREATIVE


def test_derive_seed_is_stable_and_input_keyed() -> None:
    # Same input → same seed (reproducible); different input → different seed.
    a = derive_seed("a bio", "context")
    assert a == derive_seed("a bio", "context")
    assert a != derive_seed("a different bio", "context")
    assert a != derive_seed("a bio", "other context")
    # Positive 31-bit int — the safe Ollama seed range.
    assert 0 <= a <= 0x7FFFFFFF


def test_no_anthropic_key_falls_back_to_local() -> None:
    """When the client has no Anthropic key (available_cloud_providers is
    empty), a cloud preferred model with a local fallback should promote
    the fallback."""
    d = decide(
        sensitivity=Sensitivity.PUBLIC,
        model_requested=None,
        allow_cloud_for_internal=True,
        rule=_rule(preferred="claude-sonnet-4-5", fallback="llama3.3:70b"),
        default_model="llama3.3:70b",
        default_sensitive_model="llama3.3:70b",
        available_cloud_providers=frozenset(),
    )
    assert d.model == "llama3.3:70b"
    assert d.provider == "ollama"
    assert "sensitivity-promoted-fallback" in d.reason


def test_no_anthropic_key_blocks_explicit_cloud_override() -> None:
    with pytest.raises(SensitivityViolation):
        decide(
            sensitivity=Sensitivity.PUBLIC,
            model_requested="claude-haiku-4-5",
            allow_cloud_for_internal=True,
            rule=_rule(sensitivity=Sensitivity.PUBLIC),
            default_model="llama3.3:70b",
            default_sensitive_model="llama3.3:70b",
            available_cloud_providers=frozenset(),
        )


def test_no_anthropic_key_and_no_local_fallback_raises() -> None:
    with pytest.raises(SensitivityViolation):
        decide(
            sensitivity=Sensitivity.PUBLIC,
            model_requested=None,
            allow_cloud_for_internal=True,
            rule=_rule(preferred="claude-sonnet-4-5", fallback=None),
            default_model="llama3.3:70b",
            default_sensitive_model="llama3.3:70b",
            available_cloud_providers=frozenset(),
        )


def test_anthropic_available_does_not_imply_bedrock() -> None:
    """A Bedrock model must NOT be routed-to just because the client has an
    Anthropic key. Each cloud provider is gated independently."""
    with pytest.raises(SensitivityViolation):
        decide(
            sensitivity=Sensitivity.PUBLIC,
            model_requested="anthropic.claude-3-5-sonnet-20241022-v2:0",
            allow_cloud_for_internal=True,
            rule=_rule(sensitivity=Sensitivity.PUBLIC),
            default_model="llama3.3:70b",
            default_sensitive_model="llama3.3:70b",
            available_cloud_providers=frozenset({"anthropic"}),
        )


def test_bedrock_available_routes_to_bedrock_model() -> None:
    d = decide(
        sensitivity=Sensitivity.PUBLIC,
        model_requested="anthropic.claude-3-5-sonnet-20241022-v2:0",
        allow_cloud_for_internal=True,
        rule=_rule(sensitivity=Sensitivity.PUBLIC),
        default_model="llama3.3:70b",
        default_sensitive_model="llama3.3:70b",
        available_cloud_providers=frozenset({"bedrock"}),
    )
    assert d.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert d.provider == "bedrock"


def test_no_bedrock_creds_promotes_local_fallback() -> None:
    """Bedrock primary + local fallback + no Bedrock available → promote
    fallback. Same shape as the Anthropic-key equivalent."""
    d = decide(
        sensitivity=Sensitivity.PUBLIC,
        model_requested=None,
        allow_cloud_for_internal=True,
        rule=_rule(
            preferred="anthropic.claude-3-haiku-20240307-v1:0",
            fallback="llama3.3:70b",
        ),
        default_model="llama3.3:70b",
        default_sensitive_model="llama3.3:70b",
        available_cloud_providers=frozenset({"anthropic"}),
    )
    assert d.model == "llama3.3:70b"
    assert d.provider == "ollama"
    assert "sensitivity-promoted-fallback" in d.reason
