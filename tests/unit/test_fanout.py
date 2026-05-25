"""Tests for fan-out validation and resident-model helpers."""

from __future__ import annotations

import pytest

from eval.fanout import FanoutValidationError, validate_fanout_targets
from providers.resident import is_resident, resident_model_names

# --- resident model parsing ------------------------------------------------


def test_resident_model_names_empty_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESIDENT_MODELS", "")  # force unset, ignoring any .env value
    from config.settings import get_settings

    get_settings.cache_clear()
    assert resident_model_names() == []


def test_resident_model_names_parses_comma_separated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESIDENT_MODELS", "qwen2.5:7b,llama3.2:3b,gemma:4b")
    from config.settings import get_settings

    get_settings.cache_clear()
    assert resident_model_names() == ["qwen2.5:7b", "llama3.2:3b", "gemma:4b"]


def test_resident_model_names_handles_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESIDENT_MODELS", " qwen2.5:7b , , llama3.2:3b ")
    from config.settings import get_settings

    get_settings.cache_clear()
    assert resident_model_names() == ["qwen2.5:7b", "llama3.2:3b"]


def test_is_resident_checks_membership(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESIDENT_MODELS", "qwen2.5:7b,llama3.2:3b")
    from config.settings import get_settings

    get_settings.cache_clear()
    assert is_resident("qwen2.5:7b") is True
    assert is_resident("llama3.3:70b") is False  # not in list
    assert is_resident("claude-haiku-4-5") is False


# --- fan-out target validation ---------------------------------------------


def test_validate_accepts_cloud_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESIDENT_MODELS", "")  # force unset, ignoring any .env value
    from config.settings import get_settings

    get_settings.cache_clear()
    # Should not raise — cloud doesn't need to be resident.
    validate_fanout_targets(["claude-haiku-4-5", "claude-sonnet-4-5"])


def test_validate_accepts_resident_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESIDENT_MODELS", "qwen2.5:7b,llama3.2:3b")
    from config.settings import get_settings

    get_settings.cache_clear()
    validate_fanout_targets(["qwen2.5:7b", "llama3.2:3b"])


def test_validate_rejects_non_resident_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESIDENT_MODELS", "qwen2.5:7b")
    from config.settings import get_settings

    get_settings.cache_clear()
    with pytest.raises(FanoutValidationError, match="non-resident local"):
        validate_fanout_targets(["llama3.3:70b"])


def test_validate_rejects_when_residents_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESIDENT_MODELS", "")  # force unset, ignoring any .env value
    from config.settings import get_settings

    get_settings.cache_clear()
    with pytest.raises(FanoutValidationError):
        validate_fanout_targets(["llama3.2:3b"])


def test_validate_mixed_cloud_and_resident(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESIDENT_MODELS", "qwen2.5:7b")
    from config.settings import get_settings

    get_settings.cache_clear()
    # Cloud + resident in the same list should both pass.
    validate_fanout_targets(["claude-haiku-4-5", "qwen2.5:7b"])


def test_validate_empty_list_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESIDENT_MODELS", "")  # force unset, ignoring any .env value
    from config.settings import get_settings

    get_settings.cache_clear()
    # No targets → no checks to fail.
    validate_fanout_targets([])
