"""Unit tests for `provider_for_model` and `is_cloud` covering Bedrock IDs."""

from __future__ import annotations

import pytest

from providers.registry import is_cloud, provider_for_model


@pytest.mark.parametrize(
    "model,expected",
    [
        # Anthropic direct API
        ("claude-haiku-4-5", "anthropic"),
        ("claude-sonnet-4-6", "anthropic"),
        ("claude-opus-4-5", "anthropic"),
        # Bedrock flat (no region prefix)
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", "bedrock"),
        ("anthropic.claude-3-haiku-20240307-v1:0", "bedrock"),
        ("meta.llama3-1-70b-instruct-v1:0", "bedrock"),
        ("mistral.mistral-large-2402-v1:0", "bedrock"),
        ("cohere.command-r-v1:0", "bedrock"),
        ("amazon.nova-lite-v1:0", "bedrock"),
        ("ai21.j2-ultra-v1", "bedrock"),
        # Bedrock inference profiles (cross-region) — covering every prefix
        # Anthropic models currently expose, plus the historical ones.
        ("us.anthropic.claude-3-5-sonnet-20241022-v2:0", "bedrock"),
        ("eu.anthropic.claude-3-haiku-20240307-v1:0", "bedrock"),
        ("au.anthropic.claude-haiku-4-5-20251001-v1:0", "bedrock"),
        ("jp.anthropic.claude-sonnet-4-6", "bedrock"),
        ("global.anthropic.claude-sonnet-4-6", "bedrock"),
        ("apac.anthropic.claude-3-5-sonnet-20241022-v2:0", "bedrock"),
        ("us-gov.anthropic.claude-3-5-sonnet-20241022-v2:0", "bedrock"),
        # The exact IDs verified live against the AWS docs on 2026-06-02
        ("anthropic.claude-sonnet-4-6", "bedrock"),
        ("anthropic.claude-haiku-4-5-20251001-v1:0", "bedrock"),
        ("us.anthropic.claude-sonnet-4-6", "bedrock"),
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "bedrock"),
        # Ollama — none of these should be misclassified as Bedrock
        ("llama3.3:70b", "ollama"),
        ("llama3.2:3b", "ollama"),
        ("gemma4:e4b", "ollama"),
        ("qwen3.5:9b", "ollama"),
        ("mistral-small3.2", "ollama"),
        ("mistral-medium-3.5", "ollama"),
    ],
)
def test_provider_for_model(model: str, expected: str) -> None:
    assert provider_for_model(model) == expected


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-haiku-4-5", True),
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", True),
        ("us.anthropic.claude-3-5-sonnet-20241022-v2:0", True),
        ("meta.llama3-1-70b-instruct-v1:0", True),
        ("llama3.3:70b", False),
        ("mistral-small3.2", False),
    ],
)
def test_is_cloud(model: str, expected: bool) -> None:
    assert is_cloud(model) is expected
