from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ProviderResponse:
    response: str
    tokens_in: int
    tokens_out: int
    cost_usd: Decimal
    latency_ms: int
    model_used: str
    provider: str


class ProviderError(Exception):
    """Raised when a provider call fails for a non-retryable reason."""


class ProviderTimeout(ProviderError):
    """Raised when a provider call exceeds its timeout."""


class ProviderRateLimit(ProviderError):
    """Raised when the provider returns a rate-limit signal."""


class ProviderModelNotLoaded(ProviderError):
    """Raised by Ollama when the requested model isn't currently in memory."""


class BaseProvider(ABC):
    name: str

    @abstractmethod
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
        """`temperature`/`seed` come from the task's sampling profile (see
        routing.engine). `seed` is honored only by providers that support it
        (Ollama); cloud providers ignore it and rely on temperature alone."""
        ...
