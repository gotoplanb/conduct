from __future__ import annotations

from providers.base import BaseProvider

# AWS Bedrock model IDs use a dotted namespace identifying the foundation-
# model vendor (anthropic.claude-..., meta.llama3-..., etc). Cross-region
# inference profiles wrap that with a leading region prefix
# (us.anthropic.claude-..., eu.anthropic.claude-...).
_BEDROCK_NAMESPACES = frozenset(
    {"anthropic", "amazon", "meta", "mistral", "cohere", "ai21", "deepseek", "stability"}
)
_INFERENCE_PROFILE_PREFIXES = frozenset({"us", "eu", "apac", "us-gov"})


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
        """Provider lookup that honors per-client secrets.

        - Anthropic (direct API): if the client has its own encrypted key,
          build a provider with that key; else fall through to the
          globally-registered Anthropic provider (used by tests with stub
          registries, or when a deployment registers a global fallback).
        - Bedrock: per-client only, no host fallback — strict tenant isolation
          mirrors the routing rule. Raises if the client has no AWS creds.
        - Everything else: fall through to the globally-registered provider.
        """
        if name == "anthropic" and getattr(client_app, "anthropic_api_key_encrypted", None):
            from providers.anthropic import AnthropicProvider  # noqa: PLC0415
            from secrets_box import decrypt  # noqa: PLC0415

            return AnthropicProvider(api_key=decrypt(client_app.anthropic_api_key_encrypted))
        if name == "bedrock":
            blob = getattr(client_app, "bedrock_creds_encrypted", None)
            if not blob:
                raise KeyError(
                    "client has no Bedrock credentials configured "
                    "(no host fallback is supported)"
                )
            import json  # noqa: PLC0415

            from providers.bedrock import BedrockProvider  # noqa: PLC0415
            from secrets_box import decrypt  # noqa: PLC0415

            creds = json.loads(decrypt(blob))
            return BedrockProvider(
                access_key_id=creds["access_key_id"],
                secret_access_key=creds["secret_access_key"],
                region=creds["region"],
            )
        return self.get(name)

    def has_for_client(self, client_app, name: str) -> bool:
        if name == "anthropic" and getattr(client_app, "anthropic_api_key_encrypted", None):
            return True
        if name == "bedrock":
            # No host fallback for Bedrock — only present when the client
            # itself has stored creds.
            return bool(getattr(client_app, "bedrock_creds_encrypted", None))
        return self.has(name)

    def for_model(self, model: str) -> BaseProvider:
        return self.get(provider_for_model(model))

    @property
    def names(self) -> list[str]:
        return list(self._providers)


def provider_for_model(model: str) -> str:
    """Map a model name to its serving provider.

    Anthropic's direct API uses `claude-<size>-<rev>` IDs. AWS Bedrock uses
    dotted-namespace IDs identifying the foundation-model vendor
    (`anthropic.claude-...`, `meta.llama3-...`, `amazon.nova-...`, etc.),
    optionally wrapped by a region inference profile prefix
    (`us.anthropic.claude-...`, `eu.anthropic.claude-...`). Anything else is
    assumed to be local Ollama.
    """
    if model.startswith("claude-"):
        return "anthropic"
    namespace = _bedrock_namespace(model)
    if namespace in _BEDROCK_NAMESPACES:
        return "bedrock"
    return "ollama"


def _bedrock_namespace(model: str) -> str | None:
    """Extract the foundation-model vendor namespace from a Bedrock model id,
    looking past an optional inference-profile prefix. Returns None if the id
    isn't dot-separated (i.e. not a Bedrock id)."""
    parts = model.split(".")
    if len(parts) < 2:
        return None
    if parts[0] in _INFERENCE_PROFILE_PREFIXES and len(parts) >= 3:
        return parts[1]
    return parts[0]


def is_cloud(model: str) -> bool:
    return provider_for_model(model) != "ollama"
