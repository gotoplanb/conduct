from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.pricing import get_pricing
from providers.anthropic import AnthropicProvider
from providers.base import ProviderError


def _fake_message(text: str, in_tokens: int, out_tokens: int) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    msg.usage = MagicMock(input_tokens=in_tokens, output_tokens=out_tokens)
    return msg


@pytest.mark.asyncio
async def test_complete_uses_pricing_registry(tmp_path, monkeypatch) -> None:
    pricing_file = tmp_path / "pricing.yaml"
    pricing_file.write_text(
        """
anthropic:
  claude-fake-isolation-only:
    input_per_1m_usd: 3.00
    output_per_1m_usd: 15.00
"""
    )
    # Reset the pricing singleton and point it at our temp file. Using a
    # model name that doesn't exist in the real config/pricing.yaml proves
    # the registry is actually reading the temp file — if the monkeypatch
    # silently became a no-op (regression of the def-time-default bug),
    # cost_usd would be 0 and this test would catch it.
    import config.pricing as pricing_mod

    monkeypatch.setattr(pricing_mod, "_registry", None)
    monkeypatch.setattr(pricing_mod, "DEFAULT_PRICING_PATH", pricing_file)

    provider = AnthropicProvider(api_key="sk-test")
    fake = _fake_message("hello", in_tokens=1000, out_tokens=500)
    with patch.object(provider._client.messages, "create", AsyncMock(return_value=fake)):
        result = await provider.complete(
            prompt="hi", model="claude-fake-isolation-only", system_prompt="be brief"
        )

    assert result.response == "hello"
    assert result.tokens_in == 1000
    assert result.tokens_out == 500
    # 1000 in × $3/1M + 500 out × $15/1M = $0.003 + $0.0075 = $0.0105
    assert result.cost_usd == Decimal("0.010500")
    assert result.provider == "anthropic"


@pytest.mark.asyncio
async def test_temperature_passed_seed_ignored() -> None:
    """A deterministic task sends temperature 0 to the Anthropic API — via
    extra_body since SDK 1.0 removed the typed param. There's no seed param,
    so `seed` is dropped (temperature-0 best-effort)."""
    provider = AnthropicProvider(api_key="sk-test")
    fake = _fake_message("x", 1, 1)
    create = AsyncMock(return_value=fake)
    with patch.object(provider._client.messages, "create", create):
        await provider.complete(
            prompt="hi", model="claude-haiku-4-5", max_tokens=64, temperature=0.0, seed=7
        )
    sent = create.call_args.kwargs
    assert sent["extra_body"] == {"temperature": 0.0}
    assert "seed" not in sent
    assert "temperature" not in sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model", ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-4-7", "claude-new-model"]
)
async def test_temperature_dropped_for_models_that_reject_it(model: str) -> None:
    """These models return a 400 on a non-default temperature, so a deterministic
    rule must still reach them, at the default temperature."""
    provider = AnthropicProvider(api_key="sk-test")
    create = AsyncMock(return_value=_fake_message("x", 1, 1))
    with patch.object(provider._client.messages, "create", create):
        await provider.complete(prompt="hi", model=model, max_tokens=64, temperature=0.0)
    assert "extra_body" not in create.call_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
async def test_no_text_is_an_error_not_an_empty_answer(stop_reason: str) -> None:
    """Thinking can use up max_tokens before any text, and a refusal has none:
    both must fail the call rather than record an empty response as a success."""
    provider = AnthropicProvider(api_key="sk-test")
    msg = _fake_message("unused", 1, 1)
    msg.content = [MagicMock(type="thinking")]
    msg.stop_reason = stop_reason
    with (
        patch.object(provider._client.messages, "create", AsyncMock(return_value=msg)),
        pytest.raises(ProviderError, match=stop_reason),
    ):
        await provider.complete(prompt="hi", model="claude-haiku-5-5", max_tokens=64)


def test_haiku_5_5_is_priced() -> None:
    """A model missing from pricing.yaml silently records $0."""
    assert get_pricing().cost("anthropic", "claude-haiku-5-5", 1_000_000, 1_000_000) == Decimal(
        "0.600000"
    )


@pytest.mark.asyncio
async def test_provider_kwargs_match_sdk_signature() -> None:
    """Signature-drift guard: every kwarg the provider builds must be a real
    parameter of the installed SDK's messages.create. The other tests mock
    create with AsyncMock, which accepts anything — this is what catches an
    SDK upgrade removing a param (as 1.0 did with `temperature`)."""
    import inspect

    from anthropic.resources.messages import AsyncMessages

    sdk_params = set(inspect.signature(AsyncMessages.create).parameters)
    provider = AnthropicProvider(api_key="sk-test")
    fake = _fake_message("x", 1, 1)
    create = AsyncMock(return_value=fake)
    with patch.object(provider._client.messages, "create", create):
        await provider.complete(
            prompt="hi", model="claude-haiku-4-5", system_prompt="s",
            max_tokens=64, temperature=0.7, seed=1,
        )
    sent = set(create.call_args.kwargs)
    unknown = sent - sdk_params
    assert not unknown, f"provider sends kwargs the SDK no longer accepts: {unknown}"


@pytest.mark.asyncio
async def test_temperature_omitted_when_none() -> None:
    """No profile temperature → don't send the param (provider default)."""
    provider = AnthropicProvider(api_key="sk-test")
    fake = _fake_message("x", 1, 1)
    create = AsyncMock(return_value=fake)
    with patch.object(provider._client.messages, "create", create):
        await provider.complete(prompt="hi", model="claude-x", max_tokens=64)
    assert "temperature" not in create.call_args.kwargs
