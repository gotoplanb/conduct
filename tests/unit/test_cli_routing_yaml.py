"""Round-trip tests for the routing YAML helpers in the `conduct` CLI."""

from __future__ import annotations

import pytest
import yaml

from scripts.cli import _rule_to_yaml, _yaml_to_rule_body


def test_rule_to_yaml_strips_non_editable_fields() -> None:
    rule = {
        "task_type": "x",
        "preferred_model": "llama3.3:70b",
        "fallback_model": "claude-haiku-4-5",
        "sensitivity": "internal",
        "max_tokens": 1000,
        "notes": "hello",
        "eval_shadow_models": [{"model": "qwen3.5:9b", "rate": 0.5}],
        "updated_at": "2026-05-30T00:00:00Z",
    }
    text = _rule_to_yaml(rule)
    assert "task_type" not in text
    assert "updated_at" not in text
    parsed = yaml.safe_load(text)
    assert parsed["preferred_model"] == "llama3.3:70b"
    assert parsed["eval_shadow_models"][0]["rate"] == 0.5


def test_yaml_to_rule_body_keeps_only_editable_keys() -> None:
    text = """
preferred_model: gemma4:e4b
fallback_model: claude-haiku-4-5
sensitivity: public
max_tokens: 500
notes: ""
eval_shadow_models: []
task_type: leaked-from-yaml  # should be ignored — task_type is in the URL
updated_at: 2026-01-01T00:00:00Z
"""
    body = _yaml_to_rule_body(text)
    assert body["preferred_model"] == "gemma4:e4b"
    assert "task_type" not in body
    assert "updated_at" not in body


def test_yaml_to_rule_body_rejects_non_mapping() -> None:
    with pytest.raises(ValueError):
        _yaml_to_rule_body("- just\n- a list\n")


def test_yaml_to_rule_body_handles_empty_input() -> None:
    """An empty YAML doc is a valid no-op mapping; the route will 422 it."""
    assert _yaml_to_rule_body("") == {}


def test_roundtrip_preserves_editable_fields() -> None:
    rule = {
        "preferred_model": "p",
        "fallback_model": "f",
        "sensitivity": "confidential",
        "max_tokens": 800,
        "notes": "n",
        "eval_shadow_models": [
            {"model": "s1", "rate": 0.1, "daily_cost_cap_usd": 0.5},
        ],
    }
    body = _yaml_to_rule_body(_rule_to_yaml(rule))
    assert body == rule
