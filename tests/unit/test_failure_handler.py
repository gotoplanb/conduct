"""StaticFailureHandler decision logic + TriageFailureHandler stub guard."""

from __future__ import annotations

import pytest

from models.types import Sensitivity
from retry.base import FailureContext, HandlerAction
from retry.static import StaticFailureHandler
from retry.triage import TriageFailureHandler
from routing.engine import RoutingDecision


def _decision(
    *,
    model: str = "llama3.3:70b",
    provider: str = "ollama",
    fallback_model: str | None = "claude-sonnet-4-5",
    fallback_provider: str | None = "anthropic",
) -> RoutingDecision:
    return RoutingDecision(
        model=model,
        provider=provider,
        fallback_model=fallback_model,
        fallback_provider=fallback_provider,
        effective_sensitivity=Sensitivity.PUBLIC,
        max_tokens=1000,
        reason="test",
    )


def _ctx(decision: RoutingDecision, available: set[str]) -> FailureContext:
    return FailureContext(
        error_type="ProviderTimeout",
        error_message="boom",
        job_task_type="bio_generation",
        job_sensitivity="public",
        decision=decision,
        available_providers=frozenset(available),
    )


@pytest.mark.asyncio
async def test_fallback_when_available() -> None:
    h = StaticFailureHandler()
    out = await h.on_provider_error(_ctx(_decision(), {"ollama", "anthropic"}))
    assert out.action == HandlerAction.FALLBACK
    assert out.target_model == "claude-sonnet-4-5"
    assert out.target_provider == "anthropic"


@pytest.mark.asyncio
async def test_fail_when_no_fallback() -> None:
    h = StaticFailureHandler()
    out = await h.on_provider_error(
        _ctx(_decision(fallback_model=None, fallback_provider=None), {"ollama"})
    )
    assert out.action == HandlerAction.FAIL


@pytest.mark.asyncio
async def test_fail_when_fallback_provider_not_loaded() -> None:
    h = StaticFailureHandler()
    out = await h.on_provider_error(_ctx(_decision(), {"ollama"}))  # no anthropic
    assert out.action == HandlerAction.FAIL


@pytest.mark.asyncio
async def test_fail_when_fallback_is_same_as_primary() -> None:
    h = StaticFailureHandler()
    out = await h.on_provider_error(
        _ctx(
            _decision(fallback_model="llama3.3:70b", fallback_provider="ollama"),
            {"ollama"},
        )
    )
    assert out.action == HandlerAction.FAIL


@pytest.mark.asyncio
async def test_triage_handler_is_stub() -> None:
    h = TriageFailureHandler()
    with pytest.raises(NotImplementedError):
        await h.on_provider_error(_ctx(_decision(), {"ollama"}))
