"""Training-dataset export endpoints (GitHub issue #16).

Stream the scored data Conduct accumulates (mostly from the LLM-as-judge, see
docs/judging.md) as JSONL ready for SFT / DPO fine-tuning:

  GET /datasets/sft           → {prompt, system, completion, meta}
  GET /datasets/preferences   → {prompt, system, chosen, rejected, meta}

Admin-only. The shaping lives in eval/datasets.py; these handlers just stream.
"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from auth import admin_only
from db.session import get_session
from eval.datasets import iter_preferences, iter_sft

router = APIRouter(prefix="/datasets", tags=["datasets"], dependencies=[Depends(admin_only)])

_NDJSON = "application/x-ndjson"


def _stream(rows: list[dict], filename: str) -> StreamingResponse:
    def gen():
        for r in rows:
            yield json.dumps(r, ensure_ascii=False) + "\n"

    return StreamingResponse(
        gen(),
        media_type=_NDJSON,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/sft")
async def export_sft(
    session: Annotated[AsyncSession, Depends(get_session)],
    task_type: Annotated[str | None, Query()] = None,
    min_score: Annotated[float, Query(ge=1, le=5)] = 4.0,
    via: Annotated[str | None, Query(description="filter to one score source, e.g. judge")] = None,
    label_dim: Annotated[
        str | None, Query(description="score on one named dimension, e.g. correctness")
    ] = None,
    prompt_version: Annotated[int | None, Query()] = None,
    include_shadows: Annotated[bool, Query()] = False,
    limit: Annotated[int, Query(ge=1, le=10000)] = 1000,
) -> StreamingResponse:
    """JSONL of high-scored (prompt, system, completion) examples — responses
    whose average quality score (overall, or `label_dim`) is >= min_score."""
    rows = await iter_sft(
        session, task_type=task_type, min_score=min_score, via=via, label_dim=label_dim,
        prompt_version=prompt_version, include_shadows=include_shadows, limit=limit,
    )
    return _stream(rows, "conduct-sft.jsonl")


@router.get("/preferences")
async def export_preferences(
    session: Annotated[AsyncSession, Depends(get_session)],
    task_type: Annotated[str | None, Query()] = None,
    method: Annotated[str, Query(pattern="^(pairwise|score)$")] = "pairwise",
    min_gap: Annotated[float, Query(ge=0, le=4)] = 2.0,
    label_dim: Annotated[
        str | None, Query(description="(score method) which named dimension to compare on")
    ] = None,
    prompt_version: Annotated[int | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=10000)] = 1000,
) -> StreamingResponse:
    """JSONL of DPO (prompt, system, chosen, rejected) pairs. method=pairwise
    reads the pairwise judge's verdicts; method=score derives pairs from
    pointwise/panel score differentials (min_gap) on the same input — on
    `label_dim` if given."""
    rows = await iter_preferences(
        session, task_type=task_type, method=method, min_gap=min_gap, label_dim=label_dim,
        prompt_version=prompt_version, limit=limit,
    )
    return _stream(rows, "conduct-preferences.jsonl")
