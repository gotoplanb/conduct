"""Tests for the metadata score aggregator that powers /eval/compare's
avg_score column."""

from __future__ import annotations

from eval.rollup import aggregate_scores as _aggregate_metadata_scores


def test_empty_input_returns_empty() -> None:
    avgs, counts, _dims = _aggregate_metadata_scores([])
    assert avgs == {}
    assert counts == {}


def test_no_scores_in_metadata_returns_empty() -> None:
    avgs, counts, _dims = _aggregate_metadata_scores(
        [
            ("llama3.3:70b", {"prompt": {"path": "x"}}),
            ("qwen2.5:72b", None),
            ("claude-haiku-4-5", {}),
        ]
    )
    assert avgs == {}
    assert counts == {}


def test_single_score_per_model() -> None:
    avgs, counts, _dims = _aggregate_metadata_scores(
        [("llama3.3:70b", {"quality_scores": [{"score": 4}]})]
    )
    assert avgs == {"llama3.3:70b": 4.0}
    assert counts == {"llama3.3:70b": 1}


def test_aggregate_per_dimension_per_model() -> None:
    avgs, counts, dims = _aggregate_metadata_scores([
        ("gemma4:e4b", {"quality_scores": [
            {"score": 4, "scores": {"correctness": 5, "format": 3}},
            {"score": 2, "scores": {"correctness": 2, "format": 2}},
        ]}),
    ])
    assert avgs == {"gemma4:e4b": 3.0}
    assert counts == {"gemma4:e4b": 2}
    assert dims["gemma4:e4b"] == {"correctness": 3.5, "format": 2.5}


def test_multiple_scores_averaged() -> None:
    avgs, counts, _dims = _aggregate_metadata_scores(
        [
            ("llama3.3:70b", {"quality_scores": [{"score": 5}, {"score": 3}]}),
            ("llama3.3:70b", {"quality_scores": [{"score": 4}]}),
        ]
    )
    assert avgs == {"llama3.3:70b": 4.0}  # (5 + 3 + 4) / 3
    assert counts == {"llama3.3:70b": 3}


def test_scores_grouped_by_model() -> None:
    avgs, counts, _dims = _aggregate_metadata_scores(
        [
            ("llama3.3:70b", {"quality_scores": [{"score": 5}, {"score": 3}]}),
            ("qwen2.5:72b", {"quality_scores": [{"score": 4}]}),
            ("claude-haiku-4-5", {"quality_scores": [{"score": 2}, {"score": 2}]}),
        ]
    )
    assert avgs["llama3.3:70b"] == 4.0
    assert avgs["qwen2.5:72b"] == 4.0
    assert avgs["claude-haiku-4-5"] == 2.0
    assert counts == {"llama3.3:70b": 2, "qwen2.5:72b": 1, "claude-haiku-4-5": 2}


def test_invalid_score_values_skipped() -> None:
    avgs, counts, _dims = _aggregate_metadata_scores(
        [
            (
                "llama3.3:70b",
                {
                    "quality_scores": [
                        {"score": 4},
                        {"score": "not a number"},
                        {"score": None},
                        {},  # missing score field
                        {"score": 2},
                    ]
                },
            )
        ]
    )
    # Only the two valid scores count.
    assert avgs == {"llama3.3:70b": 3.0}
    assert counts == {"llama3.3:70b": 2}


def test_empty_model_string_ignored() -> None:
    """Failed-before-routing rows have model_used == "" — don't include them."""
    avgs, counts, _dims = _aggregate_metadata_scores(
        [
            ("", {"quality_scores": [{"score": 5}]}),
            ("llama3.3:70b", {"quality_scores": [{"score": 4}]}),
        ]
    )
    assert avgs == {"llama3.3:70b": 4.0}
    assert "" not in avgs


def test_score_handles_float_values() -> None:
    avgs, counts, _dims = _aggregate_metadata_scores(
        [("llama3.3:70b", {"quality_scores": [{"score": 4.5}, {"score": 3.5}]})]
    )
    assert avgs == {"llama3.3:70b": 4.0}
    assert counts == {"llama3.3:70b": 2}


def test_extra_fields_in_score_entry_ignored() -> None:
    """A real score entry has reviewer/note/at fields too — make sure we
    don't choke on them."""
    avgs, _, _ = _aggregate_metadata_scores(
        [
            (
                "llama3.3:70b",
                {
                    "quality_scores": [
                        {
                            "score": 5,
                            "reviewer": "dave",
                            "note": "good",
                            "at": "2026-05-02T10:00:00Z",
                        }
                    ]
                },
            )
        ]
    )
    assert avgs == {"llama3.3:70b": 5.0}
