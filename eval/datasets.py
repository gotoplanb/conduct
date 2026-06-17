"""Training-dataset export from scored jobs/shadows (GitHub issue #16).

Turns the scored comparisons Conduct accumulates — pointwise/panel
`quality_scores` and pairwise `pairwise_verdicts` (mostly produced by the
LLM-as-judge, see docs/judging.md) — into the two shapes fine-tuning wants:

  - SFT:        {prompt, system, completion, meta}  from high-scored responses
  - preference: {prompt, system, chosen, rejected, meta}  (DPO pairs)

These are pure query/shaping helpers; routes/datasets.py streams them as JSONL.
Conduct stays task-agnostic — it emits prompt/response/score; a tenant reshapes
or enriches with domain fields at export time.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from eval.scoring import score_state
from models.job import Job
from models.prompt import PromptVersion
from models.shadow import JobShadow

# Hard ceiling on candidate rows scanned, so an export can't run away.
_SCAN_CAP = 20000


def _pv(metadata: dict | None) -> int | None:
    """The prompt version id recorded on a job (None for request-override or
    media jobs)."""
    return ((metadata or {}).get("prompt") or {}).get("version_id")


def _scored(
    metadata: dict | None, via: str | None, label_dim: str | None = None
) -> tuple[float | None, int, dict[str, float]]:
    """(avg, count, dimensions) of a row's quality_scores, optionally filtered
    to one `via` provenance. When `label_dim` is set, avg/count are for THAT
    named dimension (#18); otherwise the overall score. `dimensions` always
    carries every named dimension's average, for the row's meta."""
    scores = (metadata or {}).get("quality_scores") or []
    if via:
        scores = [s for s in scores if s.get("via") == via]
    st = score_state(scores)
    dims = {d: round(v["avg"], 2) for d, v in st["dimensions"].items()}
    if label_dim:
        d = st["dimensions"].get(label_dim)
        return (d["avg"] if d else None), (d["count"] if d else 0), dims
    return st["avg"], st["count"], dims


async def _system_for_job(
    session: AsyncSession, job: Job | None, cache: dict[int, str]
) -> str:
    """The effective system prompt for a job: its system_prompt override, else
    the content of the prompt version it resolved to (so library-sourced jobs
    still export a usable system prompt). Cached by version id."""
    if job is None:
        return ""
    if job.system_prompt:
        return job.system_prompt
    pv = _pv(job.job_metadata)
    if pv is None:
        return ""
    if pv not in cache:
        row = await session.get(PromptVersion, pv)
        cache[pv] = row.content if row else ""
    return cache[pv]


async def _export_unit(
    session: AsyncSession, target_id: UUID, cache: dict[int, str]
) -> dict | None:
    """Normalize any Job or JobShadow id into a unit:
    {prompt, system, response, model, kind, prompt_version, sensitivity}.
    A shadow inherits prompt/system/sensitivity from its parent."""
    job = await session.get(Job, target_id)
    if job is not None:
        return {
            "prompt": job.prompt,
            "system": await _system_for_job(session, job, cache),
            "response": job.response,
            "model": job.model_used,
            "kind": "job",
            "prompt_version": _pv(job.job_metadata),
            "sensitivity": job.sensitivity,
        }
    shadow = await session.get(JobShadow, target_id)
    if shadow is not None:
        parent = await session.get(Job, shadow.parent_job_id)
        return {
            "prompt": parent.prompt if parent else "",
            "system": await _system_for_job(session, parent, cache),
            "response": shadow.response,
            "model": shadow.model,
            "kind": "shadow",
            "prompt_version": _pv(parent.job_metadata) if parent else None,
            "sensitivity": parent.sensitivity if parent else "confidential",
        }
    return None


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------


async def iter_sft(
    session: AsyncSession,
    *,
    task_type: str | None = None,
    min_score: float = 4.0,
    via: str | None = None,
    label_dim: str | None = None,
    prompt_version: int | None = None,
    include_shadows: bool = False,
    limit: int = 1000,
) -> list[dict]:
    """High-scored (prompt, system, completion) examples. A row qualifies if the
    average of its quality_scores (optionally filtered to `via`, or to a single
    `label_dim`) is >= min_score. All named dimensions ride in `meta`."""
    cache: dict[int, str] = {}
    rows: list[dict] = []

    job_stmt = (
        select(Job)
        .where(
            Job.status == "complete",
            Job.response != "",
            func.jsonb_exists(Job.job_metadata, "quality_scores"),
        )
        .order_by(Job.created_at.desc())
        .limit(_SCAN_CAP)
    )
    if task_type:
        job_stmt = job_stmt.where(Job.task_type == task_type)

    for job in (await session.scalars(job_stmt)).all():
        avg, n, dims = _scored(job.job_metadata, via, label_dim)
        if avg is None or avg < min_score:
            continue
        pv = _pv(job.job_metadata)
        if prompt_version is not None and pv != prompt_version:
            continue
        rows.append({
            "prompt": job.prompt,
            "system": await _system_for_job(session, job, cache),
            "completion": job.response,
            "meta": {
                "task_type": job.task_type, "model": job.model_used or "",
                "score": round(avg, 2), "n_scores": n, "dimensions": dims,
                "prompt_version": pv, "sensitivity": job.sensitivity,
                "source": "job", "id": str(job.id),
            },
        })
        if len(rows) >= limit:
            return rows

    if include_shadows:
        await _append_sft_shadows(
            session, rows, cache,
            task_type=task_type, min_score=min_score, via=via, label_dim=label_dim,
            prompt_version=prompt_version, limit=limit,
        )
    return rows


async def _append_sft_shadows(
    session: AsyncSession, rows: list[dict], cache: dict[int, str],
    *, task_type, min_score, via, label_dim, prompt_version, limit,
) -> None:
    stmt = (
        select(JobShadow)
        .where(
            JobShadow.status == "complete",
            JobShadow.response != "",
            func.jsonb_exists(JobShadow.shadow_metadata, "quality_scores"),
        )
        .order_by(JobShadow.created_at.desc())
        .limit(_SCAN_CAP)
    )
    for shadow in (await session.scalars(stmt)).all():
        if len(rows) >= limit:
            return
        avg, n, dims = _scored(shadow.shadow_metadata, via, label_dim)
        if avg is None or avg < min_score:
            continue
        parent = await session.get(Job, shadow.parent_job_id)
        if parent is None or (task_type and parent.task_type != task_type):
            continue
        pv = _pv(parent.job_metadata)
        if prompt_version is not None and pv != prompt_version:
            continue
        rows.append({
            "prompt": parent.prompt,
            "system": await _system_for_job(session, parent, cache),
            "completion": shadow.response,
            "meta": {
                "task_type": parent.task_type, "model": shadow.model,
                "score": round(avg, 2), "n_scores": n, "dimensions": dims,
                "prompt_version": pv, "sensitivity": parent.sensitivity,
                "source": "shadow", "id": str(shadow.id),
            },
        })


# ---------------------------------------------------------------------------
# Preference (DPO) pairs
# ---------------------------------------------------------------------------


async def iter_preferences(
    session: AsyncSession,
    *,
    task_type: str | None = None,
    method: str = "pairwise",
    min_gap: float = 2.0,
    label_dim: str | None = None,
    prompt_version: int | None = None,
    limit: int = 1000,
) -> list[dict]:
    """(prompt, system, chosen, rejected) pairs. method='pairwise' reads the
    pairwise judge's verdicts directly; method='score' derives pairs from
    pointwise/panel score differentials on the same input (on `label_dim` if
    given, else the overall score)."""
    if method == "score":
        return await _preferences_from_scores(
            session, task_type=task_type, min_gap=min_gap, label_dim=label_dim,
            prompt_version=prompt_version, limit=limit,
        )
    return await _preferences_from_pairwise(
        session, task_type=task_type, prompt_version=prompt_version, limit=limit,
    )


def _pair_row(prompt, system, chosen, rejected, meta) -> dict:
    return {
        "prompt": prompt, "system": system,
        "chosen": chosen, "rejected": rejected, "meta": meta,
    }


async def _preferences_from_pairwise(
    session: AsyncSession, *, task_type, prompt_version, limit,
) -> list[dict]:
    """Each pairwise judge win recorded a {outcome:'win'} on the chosen side and
    'loss' on the rejected. Iterating 'win' entries yields each pair exactly
    once."""
    cache: dict[int, str] = {}
    rows: list[dict] = []
    stmt = (
        select(Job)
        .where(func.jsonb_exists(Job.job_metadata, "pairwise_verdicts"))
        .order_by(Job.created_at.desc())
        .limit(_SCAN_CAP)
    )
    if task_type:
        stmt = stmt.where(Job.task_type == task_type)

    for job in (await session.scalars(stmt)).all():
        for pair in await _pairs_from_pairwise_job(session, job, prompt_version, cache):
            rows.append(pair)
            if len(rows) >= limit:
                return rows
    return rows


async def _pairs_from_pairwise_job(
    session: AsyncSession, job: Job, prompt_version: int | None, cache: dict[int, str]
) -> list[dict]:
    """The (chosen, rejected) pairs from one job's winning pairwise verdicts."""
    winner = await _export_unit(session, job.id, cache)
    if winner is None or not winner["response"]:
        return []
    if prompt_version is not None and winner["prompt_version"] != prompt_version:
        return []
    pairs: list[dict] = []
    for v in (job.job_metadata or {}).get("pairwise_verdicts") or []:
        if v.get("outcome") != "win":
            continue
        loser = await _export_unit(session, UUID(v["against_job_id"]), cache)
        if loser is None or not loser["response"]:
            continue
        pairs.append(_pair_row(
            winner["prompt"], winner["system"], winner["response"], loser["response"],
            {
                "task_type": job.task_type, "chosen_model": winner["model"],
                "rejected_model": loser["model"], "prompt_version": winner["prompt_version"],
                "sensitivity": winner["sensitivity"], "method": "pairwise",
                "judge_job_id": v.get("judge_job_id"),
            },
        ))
    return pairs


async def _preferences_from_scores(
    session: AsyncSession, *, task_type, min_gap, label_dim, prompt_version, limit,
) -> list[dict]:
    """For each parent job, gather the parent's own response + its shadows
    (all answers to the SAME input), score each, and pair the highest- vs
    lowest-scored when the gap >= min_gap. Comparable by construction (same
    input → same prompt_version)."""
    cache: dict[int, str] = {}
    rows: list[dict] = []
    stmt = (
        select(Job)
        .where(
            Job.status == "complete",
            func.jsonb_exists(Job.job_metadata, "quality_scores"),
        )
        .order_by(Job.created_at.desc())
        .limit(_SCAN_CAP)
    )
    if task_type:
        stmt = stmt.where(Job.task_type == task_type)

    for parent in (await session.scalars(stmt)).all():
        pv = _pv(parent.job_metadata)
        if prompt_version is not None and pv != prompt_version:
            continue
        cands = await _scored_candidates(session, parent, label_dim)
        if len(cands) < 2:
            continue
        cands.sort(key=lambda c: c["score"])
        low, high = cands[0], cands[-1]
        if high["score"] - low["score"] < min_gap:
            continue
        rows.append(_pair_row(
            parent.prompt, await _system_for_job(session, parent, cache),
            high["response"], low["response"],
            {
                "task_type": parent.task_type, "chosen_model": high["model"],
                "rejected_model": low["model"], "chosen_score": high["score"],
                "rejected_score": low["score"], "prompt_version": pv,
                "sensitivity": parent.sensitivity, "method": "score",
            },
        ))
        if len(rows) >= limit:
            return rows
    return rows


async def _scored_candidates(
    session: AsyncSession, parent: Job, label_dim: str | None = None
) -> list[dict]:
    """The parent + its shadows that carry a score, as {response, model, score}.
    Scores on `label_dim` if given, else overall."""
    cands: list[dict] = []
    avg, _n, _d = _scored(parent.job_metadata, None, label_dim)
    if avg is not None and parent.response:
        cands.append({"response": parent.response, "model": parent.model_used or "", "score": avg})
    shadows = (
        await session.scalars(
            select(JobShadow).where(JobShadow.parent_job_id == parent.id)
        )
    ).all()
    for sh in shadows:
        s_avg, _n2, _d2 = _scored(sh.shadow_metadata, None, label_dim)
        if s_avg is not None and sh.response:
            cands.append({"response": sh.response, "model": sh.model, "score": s_avg})
    return cands
