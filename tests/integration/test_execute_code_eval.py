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


class _ScriptedBuild:
    """Returns the compile report for a `check`/`build`, a per-test-target report
    for a `test` run (test_target None = the model's own tests), and a mutants
    report for `mutants` — so dimension scoring can be driven deterministically."""

    def __init__(
        self, compile_report: BuildReport, test_reports: dict | None = None,
        mutants_report: BuildReport | None = None, clippy_report: BuildReport | None = None,
    ) -> None:
        self._compile = compile_report
        self._tests = test_reports or {}
        self._mutants = mutants_report
        self._clippy = clippy_report
        self.calls: list = []

    async def build(self, tar_bytes, commands, *, overlay_files=None, test_target=None, **_kw):
        self.calls.append((commands, test_target))
        if "clippy" in commands:
            return self._clippy
        if "mutants" in commands:
            return self._mutants
        if "test" in commands:
            return self._tests[test_target]
        return self._compile


def _mutants_report(stdout: str) -> BuildReport:
    return BuildReport(results={"mutants": CommandResult("mutants", 0, False, 9, stdout, "")})


def _clippy_report(exit_code: int = 0, stderr: str = "") -> BuildReport:
    return BuildReport(results={"clippy": CommandResult("clippy", exit_code, False, 5, "", stderr)})


async def _seed_target_with_source(db, client_id, tmp_path, src_lib: str) -> Job:
    cargo = '[package]\nname="x"\nversion="0.1.0"\nedition="2021"\n'
    text = f"```toml Cargo.toml\n{cargo}```\n```rust src/lib.rs\n{src_lib}\n```\n"
    job = Job(
        client_app_id=client_id, task_type="code_generation", sensitivity="internal",
        prompt="p", response="...", status=JobStatus.COMPLETE.value, model_used="m",
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    meta = build_cargo_artifact(text, job_id=job.id, output_dir=tmp_path)
    job.media_url = meta["url"]
    job.job_metadata = {"artifact": meta}
    await db.commit()
    await db.refresh(job)
    return job


def _test_report(stdout: str, exit_code: int = 0) -> BuildReport:
    return BuildReport(results={"test": CommandResult("test", exit_code, False, 8, stdout, "")})


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


async def test_golden_and_property_suites_record_independent_dimensions(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    _point_output_dir(monkeypatch, tmp_path)
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id, tmp_path)
    ce = await _seed_code_eval(db_session, client.id, target.id, apply_to_target=True)
    ce.inputs = {
        **ce.inputs,
        "suites": [
            {"dimension": "golden", "files": {"tests/golden.rs": "#[test] fn g(){}"}},
            {"dimension": "property", "property": True,
             "files": {"tests/props.rs": "// proptest"}},
        ],
    }
    await db_session.commit()

    scripted = _ScriptedBuild(
        _report(0),  # compiles
        {
            "golden": _test_report("test result: ok. 4 passed; 0 failed; 0 ignored"),
            "props": _test_report(
                "test result: FAILED. 0 passed; 1 failed\nminimal failing input: n = 0",
                exit_code=101,
            ),
        },
    )
    out = await execute_code_eval_job(
        job=ce, client=client, session=db_session, build_client=scripted,
    )

    verdict = json.loads(out.response)
    # compile + golden (all pass -> 5) + property (all fail -> 1), independent.
    assert verdict["dimensions"] == {"compile": 5, "golden": 5, "property": 1}
    assert verdict["suites"]["golden"]["pass_rate"] == 1.0
    assert verdict["suites"]["property"]["counterexample"] == "n = 0"

    await db_session.refresh(target)
    qs = (target.job_metadata or {}).get("quality_scores") or []
    assert qs[-1]["scores"] == {"compile": 5, "golden": 5, "property": 1}
    assert qs[-1]["via"] == "code-eval"


async def test_suites_skipped_when_compile_fails(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    _point_output_dir(monkeypatch, tmp_path)
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id, tmp_path)
    ce = await _seed_code_eval(db_session, client.id, target.id, apply_to_target=True)
    ce.inputs = {
        **ce.inputs,
        "suites": [{"dimension": "golden", "files": {"tests/golden.rs": "x"}}],
    }
    await db_session.commit()

    # Compile fails -> no test runs; only the compile dimension is recorded.
    scripted = _ScriptedBuild(_report(101, "error[E0308]: boom"), {})
    out = await execute_code_eval_job(
        job=ce, client=client, session=db_session, build_client=scripted,
    )
    verdict = json.loads(out.response)
    assert verdict["dimensions"] == {"compile": 1}
    assert verdict["suites"] == {}
    assert all("test" not in cmds for cmds, _ in scripted.calls)


class _StubIndex:
    """Stub crates.io index: name -> versions, or None for 'not found'."""

    def __init__(self, mapping: dict) -> None:
        self._mapping = mapping

    async def versions(self, name):
        if name not in self._mapping:
            return None
        return self._mapping[name]


async def test_check_deps_flags_hallucinated_dependency(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    _point_output_dir(monkeypatch, tmp_path)
    client = seeded_client[0]
    # Target crate whose Cargo.toml references a fabricated crate.
    cargo = '[package]\nname="x"\nversion="0.1.0"\nedition="2021"\n[dependencies]\nmadeup_xyz="1"\n'
    text = f"```toml Cargo.toml\n{cargo}```\n```rust src/lib.rs\npub fn f() {{}}\n```\n"
    target = Job(
        client_app_id=client.id, task_type="code_generation", sensitivity="internal",
        prompt="p", response="...", status=JobStatus.COMPLETE.value, model_used="m",
    )
    db_session.add(target)
    await db_session.commit()
    await db_session.refresh(target)
    meta = build_cargo_artifact(text, job_id=target.id, output_dir=tmp_path)
    target.media_url = meta["url"]
    target.job_metadata = {"artifact": meta}
    await db_session.commit()

    ce = await _seed_code_eval(db_session, client.id, target.id, apply_to_target=True)
    ce.inputs = {**ce.inputs, "check_deps": True}
    await db_session.commit()

    out = await execute_code_eval_job(
        job=ce, client=client, session=db_session,
        build_client=_FakeBuild(_report(0)), index_client=_StubIndex({}),  # madeup_xyz -> not found
    )
    verdict = json.loads(out.response)
    assert verdict["dimensions"]["deps"] == 1
    assert verdict["deps"]["offenders"][0]["name"] == "madeup_xyz"
    assert verdict["deps"]["offenders"][0]["reason"] == "not_found"


async def test_check_deps_clean_when_real(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    _point_output_dir(monkeypatch, tmp_path)
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id, tmp_path)  # std-only, no deps
    ce = await _seed_code_eval(db_session, client.id, target.id, apply_to_target=False)
    ce.inputs = {**ce.inputs, "check_deps": True}
    await db_session.commit()

    out = await execute_code_eval_job(
        job=ce, client=client, session=db_session,
        build_client=_FakeBuild(_report(0)), index_client=_StubIndex({}),
    )
    verdict = json.loads(out.response)
    assert verdict["dimensions"]["deps"] == 5  # no registry deps -> trivially clean
    assert verdict["deps"]["checked"] == 0


async def test_mutation_shallow_tests_low_dimension(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    _point_output_dir(monkeypatch, tmp_path)
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id, tmp_path)
    ce = await _seed_code_eval(db_session, client.id, target.id, apply_to_target=True)
    ce.inputs = {**ce.inputs, "check_mutants": True}
    await db_session.commit()

    scripted = _ScriptedBuild(
        _report(0),
        {None: _test_report("test result: ok. 1 passed; 0 failed")},  # model tests pass
        mutants_report=_mutants_report("5 mutants tested in 0s: 5 missed"),  # all survive
    )
    out = await execute_code_eval_job(
        job=ce, client=client, session=db_session, build_client=scripted,
    )
    verdict = json.loads(out.response)
    assert verdict["dimensions"]["mutation"] == 1  # kill_rate 0 -> shallow
    assert verdict["mutation"]["missed"] == 5 and verdict["mutation"]["kill_rate"] == 0.0


async def test_mutation_thorough_tests_high_dimension(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    _point_output_dir(monkeypatch, tmp_path)
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id, tmp_path)
    ce = await _seed_code_eval(db_session, client.id, target.id, apply_to_target=False)
    ce.inputs = {**ce.inputs, "check_mutants": True}
    await db_session.commit()

    scripted = _ScriptedBuild(
        _report(0),
        {None: _test_report("test result: ok. 3 passed; 0 failed")},
        mutants_report=_mutants_report("5 mutants tested in 0s: 5 caught"),
    )
    out = await execute_code_eval_job(
        job=ce, client=client, session=db_session, build_client=scripted,
    )
    assert json.loads(out.response)["dimensions"]["mutation"] == 5  # kill_rate 1.0


async def test_mutation_skipped_when_no_model_tests(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    _point_output_dir(monkeypatch, tmp_path)
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id, tmp_path)
    ce = await _seed_code_eval(db_session, client.id, target.id, apply_to_target=False)
    ce.inputs = {**ce.inputs, "check_mutants": True}
    await db_session.commit()

    scripted = _ScriptedBuild(
        _report(0),
        {None: _test_report("test result: ok. 0 passed; 0 failed")},  # no model tests
    )
    out = await execute_code_eval_job(
        job=ce, client=client, session=db_session, build_client=scripted,
    )
    verdict = json.loads(out.response)
    assert "mutation" not in verdict["dimensions"]  # NA, not a bogus score
    assert verdict["mutation"]["skipped"] == "no model-authored tests"


async def test_structural_flags_unsafe(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    _point_output_dir(monkeypatch, tmp_path)
    client = seeded_client[0]
    target = await _seed_target_with_source(
        db_session, client.id, tmp_path, "pub fn f() { unsafe { } }"
    )
    ce = await _seed_code_eval(db_session, client.id, target.id, apply_to_target=True)
    ce.inputs = {**ce.inputs, "check_structural": True}
    await db_session.commit()

    scripted = _ScriptedBuild(_report(0), clippy_report=_clippy_report(0))  # AST runs on real tar
    out = await execute_code_eval_job(
        job=ce, client=client, session=db_session, build_client=scripted,
    )
    verdict = json.loads(out.response)
    assert verdict["dimensions"]["structural"] == 1
    assert verdict["structural"]["ast"]["unsafe"][0]["kind"] == "unsafe_block"


async def test_structural_clean_passes(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    _point_output_dir(monkeypatch, tmp_path)
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id, tmp_path)  # clean add()
    ce = await _seed_code_eval(db_session, client.id, target.id, apply_to_target=False)
    ce.inputs = {**ce.inputs, "check_structural": True}
    await db_session.commit()

    out = await execute_code_eval_job(
        job=ce, client=client, session=db_session,
        build_client=_ScriptedBuild(_report(0), clippy_report=_clippy_report(0)),
    )
    verdict = json.loads(out.response)
    assert verdict["dimensions"]["structural"] == 5
    assert verdict["structural"]["clippy"]["ok"] is True


async def test_structural_clippy_failure_flags(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    _point_output_dir(monkeypatch, tmp_path)
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id, tmp_path)  # clean AST
    ce = await _seed_code_eval(db_session, client.id, target.id, apply_to_target=False)
    ce.inputs = {**ce.inputs, "check_structural": True}
    await db_session.commit()

    # AST clean but clippy denies a lint (exit != 0) -> structural fails.
    out = await execute_code_eval_job(
        job=ce, client=client, session=db_session,
        build_client=_ScriptedBuild(
            _report(0), clippy_report=_clippy_report(101, "warning: needless return")
        ),
    )
    verdict = json.loads(out.response)
    assert verdict["dimensions"]["structural"] == 1
    assert verdict["structural"]["ast"]["ok"] is True
    assert verdict["structural"]["clippy"]["ok"] is False


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
