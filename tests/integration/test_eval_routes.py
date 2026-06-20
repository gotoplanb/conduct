"""Routes under /eval — compare, score (job + shadow), review."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from models.job import Job
from models.shadow import JobShadow
from models.types import JobStatus


@pytest.fixture
def task_type() -> str:
    """Unique task_type per test, isolated from any pre-existing rule/job."""
    return f"eval-test-{uuid4().hex[:8]}"


async def _seed_job(
    db: AsyncSession,
    *,
    client_id,
    task_type: str,
    model: str = "llama3.3:70b",
    status: str = JobStatus.COMPLETE.value,
    latency: int = 100,
    tokens_out: int = 50,
    cost: Decimal = Decimal("0"),
    metadata: dict | None = None,
) -> Job:
    job = Job(
        client_app_id=client_id,
        task_type=task_type,
        sensitivity="internal",
        prompt="x",
        status=status,
        model_used=model,
        latency_ms=latency,
        tokens_in=10,
        tokens_out=tokens_out,
        cost_usd=cost,
        completed_at=datetime.now(UTC),
        job_metadata=metadata or {},
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def test_compare_aggregates_jobs(
    client, admin_headers, db_session, seeded_client, task_type
) -> None:
    cid = seeded_client[0].id
    await _seed_job(
        db_session, client_id=cid, task_type=task_type, model="llama3.3:70b", latency=100
    )
    await _seed_job(
        db_session, client_id=cid, task_type=task_type, model="llama3.3:70b", latency=200
    )
    await _seed_job(
        db_session, client_id=cid, task_type=task_type, model="qwen2.5:7b", latency=50
    )

    r = await client.get(f"/eval/compare?task_type={task_type}", headers=admin_headers)
    assert r.status_code == 200
    by_model = {m["model"]: m for m in r.json()["models"]}
    assert by_model["llama3.3:70b"]["job_count"] == 2
    assert by_model["llama3.3:70b"]["avg_latency_ms"] == 150.0
    assert by_model["qwen2.5:7b"]["job_count"] == 1


async def test_compare_folds_in_shadows(
    client, admin_headers, db_session, seeded_client, task_type
) -> None:
    cid = seeded_client[0].id
    parent = await _seed_job(db_session, client_id=cid, task_type=task_type, model="llama3.3:70b")
    db_session.add(
        JobShadow(
            parent_job_id=parent.id,
            model="qwen2.5:7b",
            provider="ollama",
            status=JobStatus.COMPLETE.value,
            response="alt",
            latency_ms=80,
            tokens_in=10,
            tokens_out=40,
            cost_usd=Decimal("0"),
            completed_at=datetime.now(UTC),
        )
    )
    await db_session.commit()

    r = await client.get(f"/eval/compare?task_type={task_type}", headers=admin_headers)
    by_model = {m["model"]: m for m in r.json()["models"]}
    # qwen showed up as a shadow even though no real job ran on it
    assert "qwen2.5:7b" in by_model
    assert by_model["qwen2.5:7b"]["job_count"] == 1


async def test_score_job(client, admin_headers, db_session, seeded_client, task_type) -> None:
    parent = await _seed_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    r = await client.post(
        f"/eval/jobs/{parent.id}/score",
        json={"score": 4, "reviewer": "dave", "note": "good"},
        headers=admin_headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "job"
    assert body["scores"][-1]["score"] == 4


async def test_score_job_named_dimensions(
    client, admin_headers, db_session, seeded_client, task_type
) -> None:
    parent = await _seed_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    r = await client.post(
        f"/eval/jobs/{parent.id}/score",
        json={"scores": {"correctness": 5, "format": 3, "craft": 4}, "reviewer": "dave"},
        headers=admin_headers,
    )
    assert r.status_code == 200
    entry = r.json()["scores"][-1]
    assert entry["score"] == 4  # round(mean)
    assert entry["scores"] == {"correctness": 5, "format": 3, "craft": 4}


async def test_score_neither_score_nor_scores_is_400(
    client, admin_headers, db_session, seeded_client, task_type
) -> None:
    parent = await _seed_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    r = await client.post(
        f"/eval/jobs/{parent.id}/score", json={"reviewer": "x"}, headers=admin_headers
    )
    assert r.status_code == 400


async def test_score_shadow(client, admin_headers, db_session, seeded_client, task_type) -> None:
    parent = await _seed_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    shadow = JobShadow(
        parent_job_id=parent.id,
        model="qwen2.5:7b",
        provider="ollama",
        status=JobStatus.COMPLETE.value,
        response="alt",
    )
    db_session.add(shadow)
    await db_session.commit()
    await db_session.refresh(shadow)

    r = await client.post(
        f"/eval/jobs/{shadow.id}/score",
        json={"score": 3},
        headers=admin_headers,
    )
    assert r.status_code == 200
    assert r.json()["kind"] == "shadow"


async def test_score_unknown_id_404(client, admin_headers) -> None:
    r = await client.post(
        f"/eval/jobs/{uuid4()}/score",
        json={"score": 5},
        headers=admin_headers,
    )
    assert r.status_code == 404


async def test_review_returns_unscored_shadows(
    client, admin_headers, db_session, seeded_client, task_type
) -> None:
    parent = await _seed_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    unscored = JobShadow(
        parent_job_id=parent.id,
        model="qwen2.5:7b",
        provider="ollama",
        status=JobStatus.COMPLETE.value,
        response="alt-a",
    )
    scored = JobShadow(
        parent_job_id=parent.id,
        model="llama3.2:3b",
        provider="ollama",
        status=JobStatus.COMPLETE.value,
        response="alt-b",
        shadow_metadata={"quality_scores": [{"score": 4}]},
    )
    db_session.add_all([unscored, scored])
    await db_session.commit()

    r = await client.get(f"/eval/review?task_type={task_type}", headers=admin_headers)
    assert r.status_code == 200
    shadow_ids = {item["shadow_id"] for item in r.json()["items"]}
    assert str(unscored.id) in shadow_ids
    assert str(scored.id) not in shadow_ids


async def test_compare_surfaces_composite(
    client, admin_headers, db_session, seeded_client, task_type
) -> None:
    cid = seeded_client[0].id
    # A code-eval'd job: per-dimension scores in the quality lane (#18/#30).
    md = {"quality_scores": [
        {"score": 4, "scores": {"compile": 5, "golden": 4}, "via": "code-eval"},
    ]}
    await _seed_job(db_session, client_id=cid, task_type=task_type, model="m1", metadata=md)

    r = await client.get(f"/eval/compare?task_type={task_type}", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["composite_weights"]["compile"] == 3
    comp = {m["model"]: m for m in body["models"]}["m1"]["composite"]
    # (3*5 + 3*4) / (3+3) = 4.5, decomposable back to its parts
    assert comp["score"] == 4.5
    assert comp["components"]["golden"]["avg"] == 4.0


async def test_finetuned_model_rolls_up_as_own_row(
    client, admin_headers, db_session, seeded_client, task_type
) -> None:
    # A fine-tuned checkpoint (#32) is just another model_used string — it rolls
    # up as its own row, comparable head-to-head with the base model it improves.
    cid = seeded_client[0].id
    await _seed_job(db_session, client_id=cid, task_type=task_type, model="gemma4:e4b")
    await _seed_job(db_session, client_id=cid, task_type=task_type, model="code-gen-dpo:v2")

    r = await client.get(f"/eval/compare?task_type={task_type}", headers=admin_headers)
    assert r.status_code == 200
    models = {m["model"] for m in r.json()["models"]}
    assert {"gemma4:e4b", "code-gen-dpo:v2"} <= models


async def test_eval_requires_admin(client) -> None:
    r = await client.get("/eval/compare?task_type=x")
    assert r.status_code == 403
