"""Shared quality-scoring core for jobs and shadows.

Both the JSON API (`POST /eval/jobs/{id}/score`) and the UI review page append
1-5 quality ratings to a target's metadata under `quality_scores`. The rollup
(`eval/rollup.py`) reads that slot to compute per-model average score. Keeping
the append logic here means the two entry points can't drift apart.
"""

from __future__ import annotations

import contextlib
import hashlib
import secrets
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from models.job import Job
from models.shadow import JobShadow

EVAL_TOKEN_PREFIX = "cdt_ev_"


class EvalTokenError(Exception):
    """Raised when an eval token is invalid, expired, or already used. `status`
    is the HTTP status the route should return."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def _normalize_scores(
    score: int | None, scores: dict | None
) -> tuple[int, dict[str, int] | None]:
    """Resolve a single int and/or a named-dimension map into
    (overall, dimensions). A bare int → (int, None). A dimension map (e.g.
    {"correctness": 5, "format": 3}) → (round(mean), map). Raises ValueError
    on an out-of-range value or when neither is given. Conduct doesn't know
    what the dimension names mean — they're the caller's vocabulary (#18)."""
    if scores:
        dims = {str(k): int(v) for k, v in scores.items()}
        for v in dims.values():
            if not 1 <= v <= 5:
                raise ValueError(f"dimension score {v} out of range 1-5")
        overall = round(sum(dims.values()) / len(dims))
        return overall, dims
    if score is not None:
        s = int(score)
        if not 1 <= s <= 5:
            raise ValueError(f"score {s} out of range 1-5")
        return s, None
    raise ValueError("a score or a non-empty scores map is required")


def _score_entry(
    score: int, reviewer: str | None, note: str | None,
    via: str | None = None, scores: dict | None = None,
) -> dict:
    entry = {
        "score": score,
        "reviewer": reviewer or "",
        "note": note or "",
        "via": via or "",
        "at": datetime.now(UTC).isoformat(),
    }
    if scores:
        entry["scores"] = scores
    return entry


async def apply_score(
    session: AsyncSession,
    target_id: UUID,
    *,
    score: int | None = None,
    scores: dict | None = None,
    reviewer: str | None = None,
    note: str | None = None,
    via: str | None = None,
) -> tuple[str, list[dict]] | None:
    """Append a quality score to a Job or JobShadow, by id. Jobs are tried
    first, then shadows (UUIDs don't collide across the two tables). `via` tags
    the score's provenance (e.g. "admin", "mcp", "url", "judge"). Pass `score`
    (a 1-5 int) and/or `scores` (a named-dimension map, #18). Returns
    (kind, full_scores_list) or None if no row matches."""
    overall, dims = _normalize_scores(score, scores)
    entry = _score_entry(overall, reviewer, note, via, dims)

    job = await session.get(Job, target_id)
    if job is not None:
        scores = [*(job.job_metadata or {}).get("quality_scores", []), entry]
        job.job_metadata = {**(job.job_metadata or {}), "quality_scores": scores}
        await session.commit()
        return "job", scores

    shadow = await session.get(JobShadow, target_id)
    if shadow is not None:
        scores = [*(shadow.shadow_metadata or {}).get("quality_scores", []), entry]
        shadow.shadow_metadata = {**(shadow.shadow_metadata or {}), "quality_scores": scores}
        await session.commit()
        return "shadow", scores

    return None


async def apply_pairwise_preference(
    session: AsyncSession,
    target_id: UUID,
    *,
    against_job_id: UUID,
    outcome: str,
    judge_job_id: UUID,
) -> bool:
    """Append a pairwise preference record to a Job or JobShadow's metadata
    under `pairwise_verdicts` (a list distinct from the 1-5 `quality_scores`
    lane, so relative win/loss judgements don't pollute pointwise averages).
    Used by the pairwise judge; consumed by the DPO export (#16). Returns True
    if a row matched."""
    entry = {
        "against_job_id": str(against_job_id),
        "outcome": outcome,  # "win" | "loss"
        "judge_job_id": str(judge_job_id),
        "at": datetime.now(UTC).isoformat(),
    }

    job = await session.get(Job, target_id)
    if job is not None:
        prefs = [*(job.job_metadata or {}).get("pairwise_verdicts", []), entry]
        job.job_metadata = {**(job.job_metadata or {}), "pairwise_verdicts": prefs}
        await session.commit()
        return True

    shadow = await session.get(JobShadow, target_id)
    if shadow is not None:
        prefs = [*(shadow.shadow_metadata or {}).get("pairwise_verdicts", []), entry]
        shadow.shadow_metadata = {
            **(shadow.shadow_metadata or {}),
            "pairwise_verdicts": prefs,
        }
        await session.commit()
        return True

    return False


def score_state(scores: list[dict]) -> dict:
    """Reduce a quality_scores list to {count, avg, dimensions} for display.
    `avg`/`count` are over the overall score (back-compat); `dimensions` maps
    each named dimension to its own {count, avg} (#18)."""
    vals: list[float] = []
    dim_vals: dict[str, list[float]] = defaultdict(list)
    for s in scores or []:
        with contextlib.suppress(TypeError, ValueError):
            vals.append(float(s.get("score")))
        for dim, v in (s.get("scores") or {}).items():
            try:
                dim_vals[dim].append(float(v))
            except (TypeError, ValueError):
                continue
    return {
        "count": len(vals),
        "avg": (sum(vals) / len(vals)) if vals else None,
        "dimensions": {
            d: {"count": len(vs), "avg": sum(vs) / len(vs)} for d, vs in dim_vals.items()
        },
    }


# --- single-use eval tokens (credential-less scoring links) ---


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def mint_eval_token(session: AsyncSession, job: Job) -> tuple[str, datetime]:
    """Mint (or replace) a single-use eval token for a job. Returns the raw
    token (shown once) and its expiry. Only the hash is stored."""
    raw = f"{EVAL_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    expires = datetime.now(UTC) + timedelta(days=get_settings().eval_token_ttl_days)
    job.eval_token_hash = _hash_token(raw)
    job.eval_token_expires_at = expires
    job.eval_token_used = False
    await session.commit()
    return raw, expires


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


async def redeem_eval_token(
    session: AsyncSession, job: Job, *, raw_token: str, score: int, note: str | None
) -> list[dict]:
    """Validate a job's eval token and append the score. Single-use: marks the
    token used. Raises EvalTokenError on any mismatch."""
    if not job.eval_token_hash:
        raise EvalTokenError("no eval token issued for this job", 404)
    if job.eval_token_used:
        raise EvalTokenError("eval token already used", 409)
    if job.eval_token_expires_at and _aware(job.eval_token_expires_at) < datetime.now(UTC):
        raise EvalTokenError("eval token expired", 401)
    if not secrets.compare_digest(job.eval_token_hash, _hash_token(raw_token)):
        raise EvalTokenError("invalid eval token", 401)

    entry = _score_entry(score, reviewer="link", note=note, via="url")
    scores = [*(job.job_metadata or {}).get("quality_scores", []), entry]
    job.job_metadata = {**(job.job_metadata or {}), "quality_scores": scores}
    job.eval_token_used = True
    await session.commit()
    return scores
