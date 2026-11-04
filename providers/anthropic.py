from __future__ import annotations

import time

from anthropic import APIStatusError, AsyncAnthropic, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.pricing import get_pricing
from observability.metrics import record_retry
from providers.base import (
    BaseProvider,
    ProviderError,
    ProviderRateLimit,
    ProviderResponse,
)

# The only direct-API models that take a non-default temperature. Newer ones
# (Opus 4.7+, Sonnet 5+, Haiku 5.5) return a 400, so a model not listed here
# samples at its default and a new model works without touching this list.
_TEMPERATURE_MODEL_PREFIXES = (
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-opus-4-6",
)


def _record_anthropic_retry(retry_state) -> None:  # type: ignore[no-untyped-def]
    record_retry(provider="anthropic", reason="rate_limit")


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        self._client = AsyncAnthropic(api_key=api_key)

    @retry(
        retry=retry_if_exception_type(ProviderRateLimit),
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
        before_sleep=_record_anthropic_retry,
    )
    async def complete(
        self,
        prompt: str,
        model: str,
        system_prompt: str = "",
        max_tokens: int = 1000,
        temperature: float | None = None,
        seed: int | None = None,
        **kwargs: object,
    ) -> ProviderResponse:
        kwargs_for_create: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs_for_create["system"] = system_prompt
        # The Anthropic API has no seed param — `seed` is ignored here, so a
        # "deterministic" task gets temperature 0 (most-stable, but not a
        # reproducibility guarantee). The temperature scale is [0, 1], which
        # matches every sampling profile's value. SDK 1.0 removed temperature
        # as a typed parameter, so it rides in extra_body, and only for the
        # models that still accept it: on the rest even "deterministic" runs
        # at the default temperature, so repeat runs can differ.
        if temperature is not None and model.startswith(_TEMPERATURE_MODEL_PREFIXES):
            kwargs_for_create["extra_body"] = {"temperature": temperature}

        started = time.perf_counter()
        try:
            msg = await self._client.messages.create(**kwargs_for_create)
        except RateLimitError as e:
            raise ProviderRateLimit(str(e)) from e
        except APIStatusError as e:
            raise ProviderError(f"anthropic {e.status_code}: {e.message}") from e
        latency_ms = int((time.perf_counter() - started) * 1000)

        text_parts = [block.text for block in msg.content if block.type == "text"]
        text = "".join(text_parts)
        # Models that think by default can spend max_tokens before any answer,
        # and a refusal has no text: neither is a successful empty response.
        if not text_parts:
            raise ProviderError(f"anthropic returned no text (stop_reason={msg.stop_reason})")
        tokens_in = msg.usage.input_tokens
        tokens_out = msg.usage.output_tokens

        return ProviderResponse(
            response=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=get_pricing().cost(self.name, model, tokens_in, tokens_out),
            latency_ms=latency_ms,
            model_used=model,
            provider=self.name,
        )
