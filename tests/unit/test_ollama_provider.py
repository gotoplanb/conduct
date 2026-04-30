from decimal import Decimal

import httpx
import pytest
import respx

from providers.base import ProviderError, ProviderModelNotLoaded
from providers.ollama import OllamaProvider


@pytest.mark.asyncio
async def test_complete_success_parses_tokens_and_zero_cost() -> None:
    base = "http://localhost:11434"
    with respx.mock(base_url=base) as mock:
        mock.post("/api/generate").mock(
            return_value=httpx.Response(
                200,
                json={
                    "response": "hello world",
                    "prompt_eval_count": 12,
                    "eval_count": 34,
                    "done": True,
                },
            )
        )
        provider = OllamaProvider(base_url=base)
        result = await provider.complete(prompt="hi", model="llama3.3:70b")

    assert result.response == "hello world"
    assert result.tokens_in == 12
    assert result.tokens_out == 34
    assert result.cost_usd == Decimal("0")
    assert result.provider == "ollama"
    assert result.model_used == "llama3.3:70b"
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_404_raises_model_not_loaded() -> None:
    base = "http://localhost:11434"
    with respx.mock(base_url=base) as mock:
        mock.post("/api/generate").mock(
            return_value=httpx.Response(404, json={"error": "model not found"})
        )
        provider = OllamaProvider(base_url=base)
        with pytest.raises(ProviderModelNotLoaded):
            await provider.complete(prompt="hi", model="missing:1b")


@pytest.mark.asyncio
async def test_500_raises_provider_error() -> None:
    base = "http://localhost:11434"
    with respx.mock(base_url=base) as mock:
        mock.post("/api/generate").mock(return_value=httpx.Response(500, text="boom"))
        provider = OllamaProvider(base_url=base)
        with pytest.raises(ProviderError):
            await provider.complete(prompt="hi", model="llama3.3:70b")
