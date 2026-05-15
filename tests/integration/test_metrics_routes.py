"""Routes under /metrics + /metrics/prometheus."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from models.job import Job


async def _seed_job(db, *, client_id, status="complete", model="llama3.3:70b", task="x"):
    job = Job(
        client_app_id=client_id,
        task_type=task,
        sensitivity="internal",
        prompt="p",
        status=status,
        model_used=model,
        latency_ms=100,
        tokens_in=10,
        tokens_out=20,
        cost_usd=Decimal("0"),
        completed_at=datetime.now(UTC),
    )
    db.add(job)
    await db.commit()
    return job


async def test_metrics_json_aggregates(client, admin_headers, db_session, seeded_client) -> None:
    await _seed_job(db_session, client_id=seeded_client[0].id, model="llama3.3:70b")
    await _seed_job(db_session, client_id=seeded_client[0].id, model="llama3.3:70b")
    await _seed_job(db_session, client_id=seeded_client[0].id, model="qwen2.5:7b")

    r = await client.get("/metrics", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["total_jobs"] >= 3
    assert "llama3.3:70b" in body["jobs_by_model"]


async def test_metrics_filters_by_task_type(
    client, admin_headers, db_session, seeded_client
) -> None:
    await _seed_job(db_session, client_id=seeded_client[0].id, task="alpha")
    await _seed_job(db_session, client_id=seeded_client[0].id, task="beta")

    r = await client.get("/metrics?task_type=alpha", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    # Only alpha jobs counted in the by-task-type rollup
    assert "alpha" in body["jobs_by_task_type"]
    assert "beta" not in body["jobs_by_task_type"]


async def test_metrics_prometheus_is_open(client) -> None:
    r = await client.get("/metrics/prometheus")
    assert r.status_code == 200
    # Prometheus text format
    assert "# HELP" in r.text or "# TYPE" in r.text or "python_" in r.text


async def test_metrics_json_requires_admin(client) -> None:
    r = await client.get("/metrics")
    assert r.status_code == 403
