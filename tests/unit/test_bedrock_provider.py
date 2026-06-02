"""Unit tests for BedrockProvider.

aioboto3's client is mocked at the session level via an async context manager
that returns a stub with a `converse` coroutine. We assert on the request shape
(Converse API envelope) and that the response is normalized into Conduct's
ProviderResponse, including pricing lookup.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from providers.base import ProviderError, ProviderRateLimit


class _FakeAsyncBedrockClient:
    """Async-context-manager stub matching aioboto3.Session().client(...)."""

    def __init__(self, converse_result=None, raise_exc=None):
        self._result = converse_result
        self._raise = raise_exc
        self.last_call_kwargs: dict | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def converse(self, **kwargs):
        self.last_call_kwargs = kwargs
        if self._raise is not None:
            raise self._raise
        return self._result


def _fake_converse_response(text: str, tokens_in: int, tokens_out: int) -> dict:
    return {
        "output": {
            "message": {
                "content": [{"text": text}],
                "role": "assistant",
            }
        },
        "usage": {"inputTokens": tokens_in, "outputTokens": tokens_out},
        "stopReason": "end_turn",
    }


def _patch_session(monkeypatch, fake_client) -> None:
    """Patch aioboto3.Session so .client(...) returns our fake context manager."""
    import providers.bedrock as br

    fake_session = MagicMock()
    fake_session.client = MagicMock(return_value=fake_client)
    monkeypatch.setattr(br.aioboto3, "Session", MagicMock(return_value=fake_session))


@pytest.mark.asyncio
async def test_complete_builds_converse_envelope_and_normalizes_response(
    tmp_path, monkeypatch
) -> None:
    pricing_file = tmp_path / "pricing.yaml"
    pricing_file.write_text(
        """
bedrock:
  anthropic.claude-3-haiku-fake-v1:0:
    input_per_1m_usd: 0.25
    output_per_1m_usd: 1.25
"""
    )
    import config.pricing as pricing_mod

    monkeypatch.setattr(pricing_mod, "_registry", None)
    monkeypatch.setattr(pricing_mod, "DEFAULT_PRICING_PATH", pricing_file)

    fake = _FakeAsyncBedrockClient(
        converse_result=_fake_converse_response("hello world", tokens_in=2000, tokens_out=500)
    )
    _patch_session(monkeypatch, fake)

    from providers.bedrock import BedrockProvider

    provider = BedrockProvider(
        access_key_id="AKIA-x", secret_access_key="sk-x", region="us-west-2"
    )
    result = await provider.complete(
        prompt="hi there",
        model="anthropic.claude-3-haiku-fake-v1:0",
        system_prompt="be brief",
        max_tokens=300,
    )

    assert result.response == "hello world"
    assert result.tokens_in == 2000
    assert result.tokens_out == 500
    # 2000 in × $0.25/1M + 500 out × $1.25/1M = $0.000500 + $0.000625 = $0.001125
    assert result.cost_usd == Decimal("0.001125")
    assert result.model_used == "anthropic.claude-3-haiku-fake-v1:0"
    assert result.provider == "bedrock"

    sent = fake.last_call_kwargs
    assert sent["modelId"] == "anthropic.claude-3-haiku-fake-v1:0"
    assert sent["messages"] == [{"role": "user", "content": [{"text": "hi there"}]}]
    assert sent["system"] == [{"text": "be brief"}]
    assert sent["inferenceConfig"] == {"maxTokens": 300}


@pytest.mark.asyncio
async def test_complete_omits_system_when_blank(monkeypatch) -> None:
    fake = _FakeAsyncBedrockClient(
        converse_result=_fake_converse_response("yo", 1, 1)
    )
    _patch_session(monkeypatch, fake)

    from providers.bedrock import BedrockProvider

    provider = BedrockProvider(
        access_key_id="x", secret_access_key="x", region="us-east-1"
    )
    await provider.complete(
        prompt="hi", model="anthropic.claude-3-haiku-fake-v1:0", max_tokens=10
    )
    assert "system" not in fake.last_call_kwargs


@pytest.mark.asyncio
async def test_complete_maps_throttling_to_rate_limit(monkeypatch) -> None:
    from botocore.exceptions import ClientError

    err = ClientError(
        error_response={"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
        operation_name="Converse",
    )
    fake = _FakeAsyncBedrockClient(raise_exc=err)
    _patch_session(monkeypatch, fake)

    from providers.bedrock import BedrockProvider

    provider = BedrockProvider(
        access_key_id="x", secret_access_key="x", region="us-east-1"
    )
    # tenacity will retry on ProviderRateLimit; we make complete fail-fast by
    # patching the retry mechanism to one attempt.
    provider.complete.retry.stop = lambda _state: True  # type: ignore[attr-defined]
    with pytest.raises(ProviderRateLimit):
        await provider.complete(
            prompt="hi", model="anthropic.claude-3-haiku-fake-v1:0", max_tokens=10
        )


@pytest.mark.asyncio
async def test_complete_maps_other_client_errors_to_provider_error(monkeypatch) -> None:
    from botocore.exceptions import ClientError

    err = ClientError(
        error_response={"Error": {"Code": "AccessDeniedException", "Message": "nope"}},
        operation_name="Converse",
    )
    fake = _FakeAsyncBedrockClient(raise_exc=err)
    _patch_session(monkeypatch, fake)

    from providers.bedrock import BedrockProvider

    provider = BedrockProvider(
        access_key_id="x", secret_access_key="x", region="us-east-1"
    )
    with pytest.raises(ProviderError, match="AccessDeniedException"):
        await provider.complete(
            prompt="hi", model="anthropic.claude-3-haiku-fake-v1:0", max_tokens=10
        )


# --- Bearer-token (long-term API key) auth ---


@pytest.mark.asyncio
async def test_complete_with_bearer_token_registers_event_hook(monkeypatch) -> None:
    """When constructed with bearer_token, the provider should register a
    before-send hook that overwrites Authorization with `Bearer <token>` and
    strips SigV4-specific headers — without touching the env var."""
    import os

    fake = _FakeAsyncBedrockClient(
        converse_result=_fake_converse_response("hi", 1, 1)
    )
    # Track event-hook registrations.
    registered: list[tuple[str, object]] = []
    fake.meta = MagicMock()
    fake.meta.events = MagicMock()
    fake.meta.events.register = MagicMock(
        side_effect=lambda event, handler: registered.append((event, handler))
    )
    _patch_session(monkeypatch, fake)
    # Guarantee no env var leaks in
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

    from providers.bedrock import BedrockProvider

    provider = BedrockProvider(region="us-east-1", bearer_token="ABSK-test-token")
    await provider.complete(
        prompt="hi", model="anthropic.claude-sonnet-4-6", max_tokens=10
    )

    # Hook registered for the right event
    assert len(registered) == 1
    event, handler = registered[0]
    assert event == "before-send.bedrock-runtime.*"

    # Invoke the captured handler against a stub request and confirm it
    # overwrites Authorization and removes the SigV4 stamps.
    class _Req:
        headers: dict[str, str] = {
            "Authorization": "AWS4-HMAC-SHA256 Credential=…",
            "X-Amz-Date": "20260602T130000Z",
            "X-Amz-Security-Token": "stale",
            "X-Amz-Content-SHA256": "stale",
            "Other-Header": "kept",
        }

    req = _Req()
    handler(req)
    assert req.headers["Authorization"] == "Bearer ABSK-test-token"
    assert "X-Amz-Date" not in req.headers
    assert "X-Amz-Security-Token" not in req.headers
    assert "X-Amz-Content-SHA256" not in req.headers
    assert req.headers["Other-Header"] == "kept"

    # Process-global env var must not have been touched
    assert "AWS_BEARER_TOKEN_BEDROCK" not in os.environ


def test_init_requires_some_credentials() -> None:
    from providers.bedrock import BedrockProvider

    with pytest.raises(ValueError, match="requires either"):
        BedrockProvider(region="us-east-1")


def test_init_rejects_both_auth_styles() -> None:
    from providers.bedrock import BedrockProvider

    with pytest.raises(ValueError, match="not both"):
        BedrockProvider(
            region="us-east-1",
            access_key_id="A",
            secret_access_key="B",
            bearer_token="ABSK",
        )
