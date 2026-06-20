"""Composite scoring (#30): fold the deterministic code-eval dimensions into one
tunable weighted score, decomposable back into its parts.

Weights are **config** (`config/composite.yaml`), not code, so "what good means"
is tunable without a redeploy. Only the configured (deterministic) dimensions
count — LLM-judge named dimensions are excluded by construction, keeping the
composite a pure deterministic signal distinct from probabilistic judgement.

The composite is computed at **read time** from the per-dimension scores already
stored on jobs (the `quality_scores` lane), so re-tuning the weights never
requires re-evaluating anything.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Relative weights across the deterministic code-eval dimensions (each scored
# 1-5). Tunable via config/composite.yaml; these are the fallback.
DEFAULT_WEIGHTS: dict[str, float] = {
    "compile": 3.0,
    "golden": 3.0,
    "property": 2.0,
    "deps": 1.0,
    "mutation": 1.0,
    "structural": 1.0,
}

_DEFAULT_PATH = "config/composite.yaml"


def load_weights(path: str | None = None) -> dict[str, float]:
    """Weights from config/composite.yaml merged over the defaults. Missing /
    malformed config falls back to DEFAULT_WEIGHTS (never raises)."""
    weights = dict(DEFAULT_WEIGHTS)
    try:
        data = yaml.safe_load(Path(path or _DEFAULT_PATH).read_text()) or {}
        for key, value in (data.get("weights") or {}).items():
            weights[str(key)] = float(value)
    except (OSError, ValueError, TypeError, yaml.YAMLError):
        pass
    return weights


def compute_composite(dim_avgs: dict, weights: dict | None = None) -> dict:
    """Weighted average of a model/job's per-dimension scores, restricted to the
    configured dimensions. Returns ``{score, components, weight_total}`` where
    `components[dim] = {avg, weight, contribution}` so the score decomposes back
    into its parts. `score` is None when no configured dimension is present."""
    weights = weights or DEFAULT_WEIGHTS
    components: dict[str, dict] = {}
    weighted_sum = weight_total = 0.0
    for dim, avg in dim_avgs.items():
        w = weights.get(dim)
        if w is None or avg is None:
            continue
        contribution = w * float(avg)
        components[dim] = {
            "avg": round(float(avg), 4), "weight": w, "contribution": round(contribution, 4),
        }
        weighted_sum += contribution
        weight_total += w
    score = round(weighted_sum / weight_total, 4) if weight_total else None
    return {"score": score, "components": components, "weight_total": weight_total}
