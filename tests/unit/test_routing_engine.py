from __future__ import annotations

import pytest

from models.routing import RoutingRule
from models.types import Sensitivity
from routing.engine import SensitivityViolation, decide


def _rule(
    task_type: str = "x",
    preferred: str = "llama3.3:70b",
    fallback: str = "claude-sonnet-4-5",
    sensitivity: Sensitivity = Sensitivity.INTERNAL,
    max_tokens: int = 1000,
) -> RoutingRule:
    r = RoutingRule(
        task_type=task_type,
        preferred_model=preferred,
        fallback_model=fallback,
        sensitivity=sensitivity.value,
        max_tokens=max_tokens,
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


def test_no_anthropic_key_falls_back_to_local() -> None:
    """When the client has no Anthropic key (cloud_available=False), a cloud
    preferred model with a local fallback should promote the fallback."""
    d = decide(
        sensitivity=Sensitivity.PUBLIC,
        model_requested=None,
        allow_cloud_for_internal=True,
        rule=_rule(preferred="claude-sonnet-4-5", fallback="llama3.3:70b"),
        default_model="llama3.3:70b",
        default_sensitive_model="llama3.3:70b",
        cloud_available=False,
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
            cloud_available=False,
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
            cloud_available=False,
        )
