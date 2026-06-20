"""Shared per-model rollup used by /eval/compare and the UI's /ui/eval page.

Previously this logic lived in two places — the JSON route and the UI route
each ran the same union-of-jobs-and-shadows + score aggregation, just with
slightly different output shapes. Consolidating into one helper keeps them
honest about agreeing.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from eval.composite import compute_composite, load_weights
from models.job import Job
from models.shadow import JobShadow
from models.types import JobStatus


def _empty_accumulator() -> dict:
    return {
        "attempts": 0,
        "successes": 0,
        "failures": 0,
        "latency_sum": 0.0,
        "latency_count": 0,
        "tokens_sum": 0.0,
        "tokens_count": 0,
        "cost_total": 0.0,
    }


def _add_row_to_accumulator(entry: dict, row: tuple) -> None:
    _, attempts, successes, failures, avg_latency, avg_tok_out, total_cost = row
    attempts_i = int(attempts or 0)
    successes_i = int(successes or 0)
    entry["attempts"] += attempts_i
    entry["successes"] += successes_i
    entry["failures"] += int(failures or 0)
    entry["cost_total"] += float(total_cost or 0)
    if avg_latency is not None and successes_i:
        entry["latency_sum"] += float(avg_latency) * successes_i
        entry["latency_count"] += successes_i
    if avg_tok_out is not None and successes_i:
        entry["tokens_sum"] += float(avg_tok_out) * successes_i
        entry["tokens_count"] += successes_i


def _accumulate_score(entry, model, sums, counts, dim_sums, dim_counts) -> None:
    """Fold one quality_scores entry into the running per-model sums (overall +
    per named dimension). Invalid / missing values are silently skipped."""
    try:
        sums[model] += float(entry.get("score"))
        counts[model] += 1
    except (TypeError, ValueError):
        pass
    for dim, raw in (entry.get("scores") or {}).items():
        try:
            dim_sums[model][dim] += float(raw)
            dim_counts[model][dim] += 1
        except (TypeError, ValueError):
            continue


def aggregate_scores(
    pairs: list[tuple[str, dict | None]],
) -> tuple[dict[str, float], dict[str, int], dict[str, dict[str, float]]]:
    """Walk a list of (model, metadata) pairs; return per-model average score,
    the count of scores that contributed, and per-model per-dimension averages
    (#18). Invalid / missing score entries are silently skipped."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    dim_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    dim_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for model, meta in pairs:
        if not model:
            continue
        for entry in (meta or {}).get("quality_scores", []):
            _accumulate_score(entry, model, sums, counts, dim_sums, dim_counts)
    avgs = {m: sums[m] / counts[m] for m in counts}
    dim_avgs = {
        m: {d: dim_sums[m][d] / dim_counts[m][d] for d in dim_sums[m]} for m in dim_sums
    }
    return avgs, dict(counts), dim_avgs


async def _fetch_job_rollup(session: AsyncSession, task_type: str, since: datetime) -> list[tuple]:
    job_is_complete = case((Job.status == JobStatus.COMPLETE.value, 1), else_=0)
    job_is_failed = case((Job.status == JobStatus.FAILED.value, 1), else_=0)
    return (
        await session.execute(
            select(
                Job.model_used.label("model"),
                func.count().label("attempts"),
                func.sum(job_is_complete).label("successes"),
                func.sum(job_is_failed).label("failures"),
                func.avg(Job.latency_ms).filter(Job.status == JobStatus.COMPLETE.value),
                func.avg(Job.tokens_out).filter(Job.status == JobStatus.COMPLETE.value),
                func.coalesce(
                    func.sum(Job.cost_usd).filter(Job.status == JobStatus.COMPLETE.value),
                    0,
                ),
            )
            .where(Job.task_type == task_type, Job.created_at >= since, Job.model_used != "")
            .group_by(Job.model_used)
        )
    ).all()


async def _fetch_shadow_rollup(
    session: AsyncSession, task_type: str, since: datetime
) -> list[tuple]:
    shadow_is_complete = case((JobShadow.status == JobStatus.COMPLETE.value, 1), else_=0)
    shadow_is_failed = case((JobShadow.status == JobStatus.FAILED.value, 1), else_=0)
    return (
        await session.execute(
            select(
                JobShadow.model.label("model"),
                func.count().label("attempts"),
                func.sum(shadow_is_complete).label("successes"),
                func.sum(shadow_is_failed).label("failures"),
                func.avg(JobShadow.latency_ms).filter(JobShadow.status == JobStatus.COMPLETE.value),
                func.avg(JobShadow.tokens_out).filter(JobShadow.status == JobStatus.COMPLETE.value),
                func.coalesce(
                    func.sum(JobShadow.cost_usd).filter(
                        JobShadow.status == JobStatus.COMPLETE.value
                    ),
                    0,
                ),
            )
            .join(Job, Job.id == JobShadow.parent_job_id)
            .where(Job.task_type == task_type, JobShadow.created_at >= since)
            .group_by(JobShadow.model)
        )
    ).all()


async def _fetch_score_pairs(
    session: AsyncSession, task_type: str, since: datetime
) -> list[tuple[str, dict | None]]:
    job_rows = (
        await session.execute(
            select(Job.model_used, Job.job_metadata).where(
                Job.task_type == task_type, Job.created_at >= since, Job.model_used != ""
            )
        )
    ).all()
    shadow_rows = (
        await session.execute(
            select(JobShadow.model, JobShadow.shadow_metadata)
            .join(Job, Job.id == JobShadow.parent_job_id)
            .where(Job.task_type == task_type, JobShadow.created_at >= since)
        )
    ).all()
    return [(m, meta) for m, meta in job_rows] + [(m, meta) for m, meta in shadow_rows]


async def compute_rollup(
    session: AsyncSession, *, task_type: str, days: int
) -> list[dict]:
    """Per-model rollup over both real jobs and their shadows, plus avg score.

    Returns a list of dicts (one per model) ordered by job_count desc. Caller
    maps into whatever response shape the endpoint needs.
    """
    since = datetime.now(UTC) - timedelta(days=days)

    job_rows = await _fetch_job_rollup(session, task_type, since)
    shadow_rows = await _fetch_shadow_rollup(session, task_type, since)

    rolled: dict[str, dict] = {}
    for source_rows in (job_rows, shadow_rows):
        for row in source_rows:
            model = row[0]
            entry = rolled.setdefault(model, _empty_accumulator())
            _add_row_to_accumulator(entry, row)

    score_pairs = await _fetch_score_pairs(session, task_type, since)
    avg_scores, score_counts, dim_scores = aggregate_scores(score_pairs)
    weights = load_weights()

    out: list[dict] = []
    for model, e in sorted(rolled.items(), key=lambda kv: -kv[1]["attempts"]):
        successes = e["successes"]
        out.append(
            {
                "model": model,
                "job_count": e["attempts"],
                "success_count": successes,
                "failure_count": e["failures"],
                "failure_rate": (e["failures"] / e["attempts"]) if e["attempts"] else 0.0,
                "avg_latency_ms": (
                    e["latency_sum"] / e["latency_count"] if e["latency_count"] else None
                ),
                "avg_tokens_out": (
                    e["tokens_sum"] / e["tokens_count"] if e["tokens_count"] else None
                ),
                "cost_per_job_usd": (e["cost_total"] / successes) if successes else 0.0,
                "avg_score": avg_scores.get(model),
                "score_count": score_counts.get(model, 0),
                "dimension_scores": dim_scores.get(model, {}),
                "composite": compute_composite(dim_scores.get(model, {}), weights),
            }
        )
    return out
