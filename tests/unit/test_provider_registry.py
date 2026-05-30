"""Unit tests for the per-client provider lookup (get_for_client)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from cryptography.fernet import Fernet

import secrets_box
from config.settings import get_settings
from providers.base import BaseProvider, ProviderResponse
from providers.registry import ProviderRegistry


class _NullProvider(BaseProvider):
    name = "anthropic"

    async def complete(self, **kwargs) -> ProviderResponse:  # pragma: no cover
        raise NotImplementedError


@dataclass
class _StubClient:
    anthropic_api_key_encrypted: str | None = None


@pytest.fixture
def secrets_key(monkeypatch: pytest.MonkeyPatch) -> str:
    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("CONDUCT_SECRETS_KEY", key)
    get_settings.cache_clear()
    secrets_box._fernet.cache_clear()
    yield key
    secrets_box._fernet.cache_clear()
    get_settings.cache_clear()


def test_get_for_client_uses_per_client_key(secrets_key) -> None:
    """When the client has an encrypted Anthropic key, the registry should
    build a fresh AnthropicProvider with that decrypted key, not return the
    globally-registered one."""
    reg = ProviderRegistry()
    reg.register(_NullProvider())  # global "anthropic" — should not be used

    encrypted = secrets_box.encrypt("sk-ant-per-client")
    client = _StubClient(anthropic_api_key_encrypted=encrypted)

    provider = reg.get_for_client(client, "anthropic")
    from providers.anthropic import AnthropicProvider

    assert isinstance(provider, AnthropicProvider)
    assert not isinstance(provider, _NullProvider)
    # AnthropicProvider doesn't expose the api_key — verify via the underlying
    # AsyncAnthropic client (the canonical place the key lives).
    assert provider._client.api_key == "sk-ant-per-client"


def test_get_for_client_no_key_falls_through_to_global(secrets_key) -> None:
    reg = ProviderRegistry()
    reg.register(_NullProvider())
    client = _StubClient(anthropic_api_key_encrypted=None)
    provider = reg.get_for_client(client, "anthropic")
    assert isinstance(provider, _NullProvider)


def test_has_for_client_true_with_encrypted_key() -> None:
    reg = ProviderRegistry()
    # Note: no global anthropic registered
    client = _StubClient(anthropic_api_key_encrypted="some-ciphertext")
    assert reg.has_for_client(client, "anthropic") is True


def test_has_for_client_false_without_key_and_no_global() -> None:
    reg = ProviderRegistry()
    client = _StubClient(anthropic_api_key_encrypted=None)
    assert reg.has_for_client(client, "anthropic") is False


def test_get_for_client_falls_through_for_non_anthropic_provider() -> None:
    """Per-client lookup is Anthropic-specific today; Ollama always uses the
    globally registered provider."""

    class _Ollama(BaseProvider):
        name = "ollama"

        async def complete(self, **kwargs) -> ProviderResponse:  # pragma: no cover
            raise NotImplementedError

    reg = ProviderRegistry()
    reg.register(_Ollama())
    client = _StubClient(anthropic_api_key_encrypted="ignored-for-ollama")
    provider = reg.get_for_client(client, "ollama")
    assert isinstance(provider, _Ollama)
