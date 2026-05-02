from __future__ import annotations

import time
from decimal import Decimal

import httpx

from providers.base import (
    BaseProvider,
    ProviderError,
    ProviderModelNotLoaded,
    ProviderResponse,
    ProviderTimeout,
)

DEFAULT_TIMEOUT_S = 300.0


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, base_url: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def complete(
        self,
        prompt: str,
        model: str,
        system_prompt: str = "",
        max_tokens: int = 1000,
        **kwargs: object,
    ) -> ProviderResponse:
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if system_prompt:
            payload["system"] = system_prompt

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(f"{self.base_url}/api/generate", json=payload)
        except httpx.TimeoutException as e:
            raise ProviderTimeout(f"ollama timeout after {self.timeout_s}s") from e
        except httpx.RequestError as e:
            raise ProviderError(f"ollama request failed: {e}") from e
        latency_ms = int((time.perf_counter() - started) * 1000)

        if resp.status_code == 404:
            # Ollama returns 404 with an error when the model isn't pulled or loaded.
            raise ProviderModelNotLoaded(f"model {model} not available on Ollama")
        if resp.status_code >= 400:
            raise ProviderError(f"ollama returned {resp.status_code}: {resp.text[:200]}")

        body = resp.json()
        return ProviderResponse(
            response=body.get("response", ""),
            tokens_in=int(body.get("prompt_eval_count", 0)),
            tokens_out=int(body.get("eval_count", 0)),
            cost_usd=Decimal("0"),
            latency_ms=latency_ms,
            model_used=model,
            provider=self.name,
        )

    async def list_models(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            return resp.json().get("models", [])

    async def list_loaded(self) -> list[dict]:
        """Return models currently resident in VRAM/RAM."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/api/ps")
            resp.raise_for_status()
            return resp.json().get("models", [])

    async def load(self, model: str, keep_alive: str | int | None = None) -> None:
        """Force-load a model by issuing an empty generate.

        Pass `keep_alive=-1` to pin the model in memory indefinitely (used for
        resident-model registration at worker boot).
        """
        payload: dict = {"model": model, "prompt": "", "stream": False}
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()

    async def unload(self, model: str) -> None:
        """Free a model from memory by setting keep_alive=0."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
            )
            resp.raise_for_status()
