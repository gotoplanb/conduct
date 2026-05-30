from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.anthropic import AnthropicProvider


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
