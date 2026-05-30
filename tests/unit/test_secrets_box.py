"""Unit tests for the Fernet wrapper that encrypts per-client secrets at rest."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

import secrets_box
from config.settings import get_settings


@pytest.fixture(autouse=True)
def _reset_cache():
    secrets_box._fernet.cache_clear()
    get_settings.cache_clear()
    yield
    secrets_box._fernet.cache_clear()
    get_settings.cache_clear()


def test_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONDUCT_SECRETS_KEY", Fernet.generate_key().decode("ascii"))
    token = secrets_box.encrypt("sk-ant-test-value")
    assert token != "sk-ant-test-value"
    assert secrets_box.decrypt(token) == "sk-ant-test-value"


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Empty string overrides any value in .env (pydantic-settings reads .env
    # directly, so `delenv` alone wouldn't simulate "unset" on machines where
    # the operator has CONDUCT_SECRETS_KEY in .env).
    monkeypatch.setenv("CONDUCT_SECRETS_KEY", "")
    with pytest.raises(secrets_box.SecretsKeyMissing):
        secrets_box.encrypt("anything")


def test_decrypt_with_wrong_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONDUCT_SECRETS_KEY", Fernet.generate_key().decode("ascii"))
    token = secrets_box.encrypt("payload")

    secrets_box._fernet.cache_clear()
    get_settings.cache_clear()
    monkeypatch.setenv("CONDUCT_SECRETS_KEY", Fernet.generate_key().decode("ascii"))

    with pytest.raises(secrets_box.SecretDecryptError):
        secrets_box.decrypt(token)
