"""Shared quality-scoring core for jobs and shadows.

Both the JSON API (`POST /eval/jobs/{id}/score`) and the UI review page append
1-5 quality ratings to a target's metadata under `quality_scores`. The rollup
(`eval/rollup.py`) reads that slot to compute per-model average score. Keeping
the append logic here means the two entry points can't drift apart.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from models.job import Job
from models.shadow import JobShadow


def _score_entry(score: int, reviewer: str | None, note: str | None) -> dict:
    return {
        "score": score,
        "reviewer": reviewer or "",
        "note": note or "",
        "at": datetime.now(UTC).isoformat(),
    }


async def apply_score(
    session: AsyncSession,
    target_id: UUID,
    *,
    score: int,
    reviewer: str | None = None,
    note: str | None = None,
) -> tuple[str, list[dict]] | None:
    """Append a quality score to a Job or JobShadow, by id. Jobs are tried
    first, then shadows (UUIDs don't collide across the two tables). Returns
    (kind, full_scores_list) or None if no row matches."""
    entry = _score_entry(score, reviewer, note)

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


def score_state(scores: list[dict]) -> dict:
    """Reduce a quality_scores list to {count, avg} for display."""
    vals: list[float] = []
    for s in scores or []:
        try:
            vals.append(float(s.get("score")))
        except (TypeError, ValueError):
            continue
    return {"count": len(vals), "avg": (sum(vals) / len(vals)) if vals else None}
