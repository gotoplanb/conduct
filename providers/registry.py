from __future__ import annotations

from providers.base import BaseProvider


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> BaseProvider:
        try:
            return self._providers[name]
        except KeyError as e:
            raise KeyError(f"provider not registered: {name}") from e

    def has(self, name: str) -> bool:
        return name in self._providers

    def for_model(self, model: str) -> BaseProvider:
        return self.get(provider_for_model(model))

    @property
    def names(self) -> list[str]:
        return list(self._providers)


def provider_for_model(model: str) -> str:
    """Map a model name to its serving provider.

    Anthropic models start with `claude-`. Everything else is assumed local
    Ollama. Add new provider prefixes here when they're wired up (e.g. Bedrock).
    """
    if model.startswith("claude-"):
        return "anthropic"
    return "ollama"


def is_cloud(model: str) -> bool:
    return provider_for_model(model) != "ollama"
