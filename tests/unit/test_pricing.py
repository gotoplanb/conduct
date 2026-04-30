from decimal import Decimal
from pathlib import Path

import pytest

from config.pricing import PricingRegistry


@pytest.fixture
def pricing_file(tmp_path: Path) -> Path:
    p = tmp_path / "pricing.yaml"
    p.write_text(
        """
anthropic:
  claude-sonnet-4-5:
    input_per_1m_usd: 3.00
    output_per_1m_usd: 15.00
  claude-haiku-4-5:
    input_per_1m_usd: 1.00
    output_per_1m_usd: 5.00
"""
    )
    return p


def test_unknown_model_costs_zero(pricing_file: Path) -> None:
    reg = PricingRegistry(pricing_file)
    assert reg.cost("ollama", "llama3.3:70b", 1000, 1000) == Decimal("0")
    assert reg.cost("anthropic", "not-a-real-model", 1, 1) == Decimal("0")


def test_known_model_cost_is_per_million(pricing_file: Path) -> None:
    reg = PricingRegistry(pricing_file)
    # 1M input + 1M output on Sonnet = $3 + $15 = $18.000000
    assert reg.cost("anthropic", "claude-sonnet-4-5", 1_000_000, 1_000_000) == Decimal("18.000000")
    # 1000 input + 0 output on Haiku = $0.001
    assert reg.cost("anthropic", "claude-haiku-4-5", 1000, 0) == Decimal("0.001000")


def test_reload_picks_up_changes(pricing_file: Path) -> None:
    reg = PricingRegistry(pricing_file)
    pricing_file.write_text(
        """
anthropic:
  claude-sonnet-4-5:
    input_per_1m_usd: 6.00
    output_per_1m_usd: 30.00
"""
    )
    reg.reload()
    assert reg.cost("anthropic", "claude-sonnet-4-5", 1_000_000, 0) == Decimal("6.000000")
