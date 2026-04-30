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
        **kwargs: object,
    ) -> ProviderResponse:
        kwargs_for_create: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs_for_create["system"] = system_prompt

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
