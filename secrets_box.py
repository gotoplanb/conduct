"""App-level encryption helper for per-client secrets (Anthropic keys, etc.).

Fernet (symmetric, authenticated) against a single master key loaded from
`CONDUCT_SECRETS_KEY`. Per-client encrypted blobs live as text in the DB
(`client_apps.anthropic_api_key_encrypted` today; same pattern for any
future per-client secret). Keeping encryption at this app layer — rather
than column-level pgcrypto — means swapping to a real KMS later only
touches this module.
"""

from __future__ import annotations

from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from config.settings import get_settings


class SecretsKeyMissing(RuntimeError):
    """Raised when CONDUCT_SECRETS_KEY isn't configured and a secret is being
    accessed. Surface as a 5xx — the operator forgot to set the env var."""


class SecretDecryptError(RuntimeError):
    """Raised when a stored ciphertext can't be decrypted with the current
    master key (wrong key, or the row was written with a previous key)."""


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    key = (get_settings().secrets_key or "").strip()
    if not key:
        raise SecretsKeyMissing(
            "CONDUCT_SECRETS_KEY is not set. Generate one with `python -c "
            "\"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"`"
        )
    return Fernet(key.encode("ascii"))


def encrypt(plaintext: str) -> str:
    """Encrypt a UTF-8 string, return a base64 token suitable for storing in
    a text column."""
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt(token: str) -> str:
    """Inverse of `encrypt`. Raises `SecretDecryptError` on a bad token /
    wrong key, and `SecretsKeyMissing` if the master key isn't configured."""
    try:
        return _fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as e:
        raise SecretDecryptError("ciphertext is invalid or master key mismatch") from e
