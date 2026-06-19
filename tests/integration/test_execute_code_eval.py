"""Integration tests for the code_eval evaluator (worker.executor, #25).

code_eval mirrors the judge as a submittable primitive but needs no model: it
ships a target code_generation job's stored artifact (#23) to the rust-build
sandbox and records a `compile` dimension on the target (on apply_to_target),
under a distinct via="code-eval". The build service is stubbed here; the real
toolchain is validated live against Watchtower.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from sqlalchemy.ext.asyncio import AsyncSession

import worker.executor as executor
from codegen.artifact import build_cargo_artifact
from codegen.build_client import BuildReport, BuildServiceError, CommandResult
from models.job import Job
from models.types import JobStatus
from worker.executor import execute_code_eval_job

_CARGO = '[package]\nname = "x"\nversion = "0.1.0"\nedition = "2021"\n'
_GOOD = f"```toml Cargo.toml\n{_CARGO}```\n```rust src/lib.rs\npub fn add() -> i64 {{ 0 }}\n```\n"


class _FakeBuild:
    def __init__(self, report: BuildReport | None = None, *, raise_error: bool = False) -> None:
        self._report = report
        self._raise = raise_error
        self.seen: tuple | None = None

    async def build(self, tar_bytes, commands, **_kw) -> BuildReport:
        if self._raise:
            raise BuildServiceError("build service unreachable: boom")
        self.seen = (tar_bytes, commands)
        return self._report


def _report(exit_code: int, stderr: str = "") -> BuildReport:
    return BuildReport(results={
        "check": CommandResult("check", exit_code, False, 12, "", stderr),
    })


async def _seed_target(db, client_id, tmp_path) -> Job:
    job = Job(
        client_app_id=client_id, task_type="code_generation", sensitivity="internal",
        prompt="p", response="...", status=JobStatus.COMPLETE.value, model_used="m",
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    meta = build_cargo_artifact(_GOOD, job_id=job.id, output_dir=tmp_path)
    job.media_url = meta["url"]
    job.job_metadata = {"artifact": meta}
    await db.commit()
    await db.refresh(job)
    return job


async def _seed_code_eval(db, client_id, target_id, *, apply_to_target) -> Job:
    job = Job(
        client_app_id=client_id, task_type="code_eval", sensitivity="internal",
        prompt="(code_eval)", status=JobStatus.PENDING.value,
        inputs={"target_job_id": str(target_id), "apply_to_target": apply_to_target},
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


def _point_output_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        executor, "get_settings",
        lambda: SimpleNamespace(tts_output_dir=str(tmp_path), rust_build_url="http://unused"),
    )


async def test_compile_success_records_dimension_on_target(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    _point_output_dir(monkeypatch, tmp_path)
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id, tmp_path)
    ce = await _seed_code_eval(db_session, client.id, target.id, apply_to_target=True)

    out = await execute_code_eval_job(
        job=ce, client=client, session=db_session, build_client=_FakeBuild(_report(0)),
    )

    assert out.status == JobStatus.COMPLETE.value
    verdict = json.loads(out.response)
    assert verdict["mode"] == "code_eval"
    assert verdict["dimensions"]["compile"] == 5
    assert verdict["compile"]["success"] is True
    assert verdict["applied_to_target"] is True
    assert out.job_metadata["code_eval"]["dimensions"]["compile"] == 5

    # The compile dimension landed on the TARGET, via the deterministic lane.
    await db_session.refresh(target)
    qs = (target.job_metadata or {}).get("quality_scores") or []
    assert len(qs) == 1
    assert qs[0]["scores"] == {"compile": 5}
    assert qs[0]["via"] == "code-eval"


async def test_compile_failure_records_failing_dimension(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    _point_output_dir(monkeypatch, tmp_path)
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id, tmp_path)
    ce = await _seed_code_eval(db_session, client.id, target.id, apply_to_target=True)

    report = _report(101, "src/lib.rs:1: error[E0308]: mismatched types\n")
    out = await execute_code_eval_job(
        job=ce, client=client, session=db_session, build_client=_FakeBuild(report),
    )

    assert out.status == JobStatus.COMPLETE.value  # the evaluator ran fine
    verdict = json.loads(out.response)
    assert verdict["dimensions"]["compile"] == 1
    assert verdict["compile"]["success"] is False
    assert any("E0308" in e for e in verdict["compile"]["errors"])
    await db_session.refresh(target)
    qs = (target.job_metadata or {}).get("quality_scores") or []
    assert qs[0]["scores"] == {"compile": 1}


async def test_verdict_only_does_not_touch_target(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    _point_output_dir(monkeypatch, tmp_path)
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id, tmp_path)
    ce = await _seed_code_eval(db_session, client.id, target.id, apply_to_target=False)

    out = await execute_code_eval_job(
        job=ce, client=client, session=db_session, build_client=_FakeBuild(_report(0)),
    )

    assert out.status == JobStatus.COMPLETE.value
    assert json.loads(out.response)["applied_to_target"] is False
    await db_session.refresh(target)
    assert (target.job_metadata or {}).get("quality_scores", []) == []


async def test_missing_target_fails_loudly(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    _point_output_dir(monkeypatch, tmp_path)
    client = seeded_client[0]
    from uuid import uuid4

    ce = await _seed_code_eval(db_session, client.id, uuid4(), apply_to_target=True)
    out = await execute_code_eval_job(
        job=ce, client=client, session=db_session, build_client=_FakeBuild(_report(0)),
    )
    assert out.status == JobStatus.FAILED.value
    assert "not found" in out.error


async def test_build_service_error_fails_loudly(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    _point_output_dir(monkeypatch, tmp_path)
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id, tmp_path)
    ce = await _seed_code_eval(db_session, client.id, target.id, apply_to_target=True)

    out = await execute_code_eval_job(
        job=ce, client=client, session=db_session, build_client=_FakeBuild(raise_error=True),
    )
    assert out.status == JobStatus.FAILED.value
    assert "code_eval:" in out.error
