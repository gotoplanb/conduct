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

    def get_for_client(self, client_app, name: str) -> BaseProvider:
        """Provider lookup that honors per-client secrets. For Anthropic, if
        the client has its own encrypted key set, builds a provider with that
        key (so each client's cost lives on their own Anthropic account); else
        falls through to the globally-registered provider (used by tests with
        stub registries, or when a deployment opts into a global fallback by
        setting ANTHROPIC_API_KEY). Raises if neither is available."""
        if name == "anthropic" and getattr(client_app, "anthropic_api_key_encrypted", None):
            from providers.anthropic import AnthropicProvider  # noqa: PLC0415
            from secrets_box import decrypt  # noqa: PLC0415

            return AnthropicProvider(api_key=decrypt(client_app.anthropic_api_key_encrypted))
        return self.get(name)

    def has_for_client(self, client_app, name: str) -> bool:
        if name == "anthropic" and getattr(client_app, "anthropic_api_key_encrypted", None):
            return True
        return self.has(name)

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
