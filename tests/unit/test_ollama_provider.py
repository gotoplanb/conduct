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
async def test_temperature_and_seed_land_in_options() -> None:
    """The sampling profile reaches Ollama via `options.temperature` and
    `options.seed` — this is what makes a deterministic task reproducible."""
    base = "http://localhost:11434"
    captured: dict = {}
    with respx.mock(base_url=base) as mock:
        def _capture(request):
            import json
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"response": "x", "done": True})

        mock.post("/api/generate").mock(side_effect=_capture)
        provider = OllamaProvider(base_url=base)
        await provider.complete(
            prompt="hi", model="m", max_tokens=50, temperature=0.0, seed=12345
        )

    assert captured["options"]["temperature"] == 0.0
    assert captured["options"]["seed"] == 12345
    assert captured["options"]["num_predict"] == 50


@pytest.mark.asyncio
async def test_options_omit_temperature_and_seed_when_unset() -> None:
    """No sampling params → don't send them; Ollama uses its own defaults and
    randomizes the seed each call (the variable-output case)."""
    base = "http://localhost:11434"
    captured: dict = {}
    with respx.mock(base_url=base) as mock:
        def _capture(request):
            import json
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"response": "x", "done": True})

        mock.post("/api/generate").mock(side_effect=_capture)
        provider = OllamaProvider(base_url=base)
        await provider.complete(prompt="hi", model="m")

    assert "temperature" not in captured["options"]
    assert "seed" not in captured["options"]


@pytest.mark.asyncio
async def test_num_ctx_lands_in_complete_options() -> None:
    """A bounded num_ctx must reach Ollama on serve, so the request reuses the
    pinned resident instance instead of reloading at the model's max context
    (which would evict the rest of the resident set)."""
    base = "http://localhost:11434"
    captured: dict = {}
    with respx.mock(base_url=base) as mock:
        def _capture(request):
            import json
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"response": "x", "done": True})

        mock.post("/api/generate").mock(side_effect=_capture)
        provider = OllamaProvider(base_url=base, num_ctx=16384)
        await provider.complete(prompt="hi", model="m")

    assert captured["options"]["num_ctx"] == 16384


@pytest.mark.asyncio
async def test_num_ctx_lands_in_load_options() -> None:
    """load() must pin at the same bounded context complete() serves at."""
    base = "http://localhost:11434"
    captured: dict = {}
    with respx.mock(base_url=base) as mock:
        def _capture(request):
            import json
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"response": "", "done": True})

        mock.post("/api/generate").mock(side_effect=_capture)
        provider = OllamaProvider(base_url=base, num_ctx=16384)
        await provider.load("m", keep_alive=-1)

    assert captured["options"]["num_ctx"] == 16384
    assert captured["keep_alive"] == -1


@pytest.mark.asyncio
async def test_num_ctx_omitted_when_unset() -> None:
    """Default (no num_ctx) must not inject the option — preserves prior behavior."""
    base = "http://localhost:11434"
    captured: dict = {}
    with respx.mock(base_url=base) as mock:
        def _capture(request):
            import json
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"response": "x", "done": True})

        mock.post("/api/generate").mock(side_effect=_capture)
        provider = OllamaProvider(base_url=base)
        await provider.complete(prompt="hi", model="m")

    assert "num_ctx" not in captured["options"]


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
