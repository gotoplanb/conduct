"""Pricing registry: loads per-model rates from config/pricing.yaml.

Hot-reloadable via SIGHUP — the FastAPI lifespan registers a handler that calls
PricingRegistry.reload(). Provider+model pairs not in the file cost zero (this
is how local providers like Ollama get $0 for free).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import yaml

DEFAULT_PRICING_PATH = Path(__file__).parent / "pricing.yaml"
COST_QUANTIZE = Decimal("0.000001")


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m_usd: Decimal
    output_per_1m_usd: Decimal


class PricingRegistry:
    def __init__(self, path: Path | None = None) -> None:
        # Resolve at call time, not def time, so tests can
        # `monkeypatch.setattr(pricing_mod, "DEFAULT_PRICING_PATH", ...)`
        # to point at a temp fixture.
        self.path = path if path is not None else DEFAULT_PRICING_PATH
        self._lock = threading.Lock()
        self._prices: dict[tuple[str, str], ModelPrice] = {}
        self.reload()

    def reload(self) -> None:
        with self.path.open() as f:
            data = yaml.safe_load(f) or {}
        new: dict[tuple[str, str], ModelPrice] = {}
        for provider, models in data.items():
            if not isinstance(models, dict):
                continue
            for model, price in models.items():
                new[(provider, model)] = ModelPrice(
                    input_per_1m_usd=Decimal(str(price["input_per_1m_usd"])),
                    output_per_1m_usd=Decimal(str(price["output_per_1m_usd"])),
                )
        with self._lock:
            self._prices = new

    def cost(self, provider: str, model: str, tokens_in: int, tokens_out: int) -> Decimal:
        with self._lock:
            price = self._prices.get((provider, model))
        if price is None:
            return Decimal("0")
        cost_in = (Decimal(tokens_in) * price.input_per_1m_usd) / Decimal(1_000_000)
        cost_out = (Decimal(tokens_out) * price.output_per_1m_usd) / Decimal(1_000_000)
        return (cost_in + cost_out).quantize(COST_QUANTIZE)

    def configured_models(self) -> list[tuple[str, str]]:
        """Return [(provider, model)] pairs that have pricing configured."""
        with self._lock:
            return sorted(self._prices.keys())


_registry: PricingRegistry | None = None


def get_pricing() -> PricingRegistry:
    global _registry
    if _registry is None:
        _registry = PricingRegistry()
    return _registry
