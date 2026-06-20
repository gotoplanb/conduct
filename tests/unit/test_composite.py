"""Unit tests for eval.composite — the deterministic composite score (#30)."""

from __future__ import annotations

from eval.composite import DEFAULT_WEIGHTS, compute_composite, load_weights


def test_weighted_average_and_decomposition() -> None:
    # compile 5 (w3) + golden 4 (w3) -> (15 + 12) / 6 = 4.5
    r = compute_composite({"compile": 5.0, "golden": 4.0}, {"compile": 3, "golden": 3})
    assert r["score"] == 4.5
    assert r["weight_total"] == 6
    assert r["components"]["compile"] == {"avg": 5.0, "weight": 3, "contribution": 15.0}
    assert r["components"]["golden"]["contribution"] == 12.0


def test_unequal_weights() -> None:
    # compile 5 (w3), deps 1 (w1) -> (15 + 1) / 4 = 4.0
    r = compute_composite({"compile": 5.0, "deps": 1.0}, {"compile": 3, "deps": 1})
    assert r["score"] == 4.0


def test_excludes_non_configured_dimensions() -> None:
    # An LLM-judge dimension ("humor") isn't in the weights -> never counts.
    r = compute_composite({"compile": 5.0, "humor": 1.0}, {"compile": 3})
    assert "humor" not in r["components"]
    assert r["score"] == 5.0


def test_missing_dimension_is_skipped_not_zeroed() -> None:
    # A model with only `compile` scored isn't dragged down by absent dims.
    r = compute_composite({"compile": 4.0})
    assert r["score"] == 4.0
    assert set(r["components"]) == {"compile"}


def test_empty_is_none() -> None:
    assert compute_composite({})["score"] is None
    assert compute_composite({"unknown": 5.0}, {"compile": 1})["score"] is None


def test_default_weights_cover_the_code_eval_dimensions() -> None:
    assert set(DEFAULT_WEIGHTS) == {
        "compile", "golden", "property", "deps", "mutation", "structural",
    }


def test_load_weights_reads_yaml_override(tmp_path) -> None:
    cfg = tmp_path / "composite.yaml"
    cfg.write_text("weights:\n  compile: 10\n  golden: 1\n")
    w = load_weights(str(cfg))
    assert w["compile"] == 10.0 and w["golden"] == 1.0
    assert w["deps"] == DEFAULT_WEIGHTS["deps"]  # unspecified -> default kept


def test_load_weights_missing_file_falls_back(tmp_path) -> None:
    assert load_weights(str(tmp_path / "nope.yaml")) == DEFAULT_WEIGHTS
