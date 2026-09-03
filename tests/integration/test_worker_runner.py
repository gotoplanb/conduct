"""Worker dispatch behavior (worker/runner.py).

_dispatch_job is the fan-in point for every task shape — tts, media,
code_eval, dpo_fine_tune, judge, and plain text — so each branch gets a
behavioral test: seed a real Job (+ rule where relevant), stub the executor
that branch hands off to, and assert the right one ran with the right
arguments. The swap/registry helpers are tested directly where dispatch
would need a live Ollama to reach them.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import worker.runner as runner
from models.job import Job
from models.routing import RoutingRule
from models.types import JobStatus

# --- fixtures ---


@pytest.fixture
def worker_sessionmaker(db_conn, monkeypatch):
    maker = async_sessionmaker(
        bind=db_conn, expire_on_commit=False,
        join_transaction_mode="create_savepoint", class_=AsyncSession,
    )
    monkeypatch.setattr(runner, "get_worker_session_maker", lambda: maker)
    return maker


@pytest.fixture
def dispatch_registry(stub_registry, monkeypatch):
    """Dispatch builds the process-global registry on first use; tests must
    never construct real providers, so swap in the stub registry."""
    monkeypatch.setattr(runner, "_get_providers", lambda: stub_registry)
    return stub_registry


@pytest.fixture
def forbid_execute_job(monkeypatch):
    """Branches that return before the plain-text path must never reach
    execute_job — make it loud if they do."""
    async def _never(**_kwargs):
        raise AssertionError("execute_job must not run for this branch")

    monkeypatch.setattr(runner, "execute_job", _never)


async def _seed_job(
    db, client_id, *, task_type, sensitivity="internal",
    status=JobStatus.PENDING.value, inputs=None, model_requested="",
    job_metadata=None,
) -> Job:
    job = Job(
        client_app_id=client_id, task_type=task_type, sensitivity=sensitivity,
        prompt="hello", status=status, inputs=inputs or {},
        model_requested=model_requested, job_metadata=job_metadata or {},
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _seed_media_rule(db, *, task_type, media_kind, preferred_model) -> RoutingRule:
    rule = RoutingRule(
        task_type=task_type, preferred_model=preferred_model,
        fallback_model="", media_kind=media_kind,
    )
    db.add(rule)
    await db.commit()
    return rule


def _task_type(prefix: str) -> str:
    # Unique per test — RoutingRule is keyed by task_type and the dev DB may
    # carry seed rules for common names.
    return f"{prefix}-{uuid4().hex[:8]}"


# --- run_job entry point ---


def test_run_job_wraps_async_dispatch(monkeypatch) -> None:
    seen: list[UUID] = []

    async def fake_run(job_id: UUID) -> None:
        seen.append(job_id)

    monkeypatch.setattr(runner, "_run_async", fake_run)
    jid = uuid4()
    runner.run_job(str(jid))  # outside RQ, rq_trace_context is a no-op
    assert seen == [jid]


# --- _dispatch_job early returns ---


async def test_dispatch_missing_job_is_noop(
    worker_sessionmaker, dispatch_registry, forbid_execute_job, caplog
) -> None:
    await runner._dispatch_job(uuid4())
    assert any("non-existent job" in r.message for r in caplog.records)


async def test_dispatch_skips_non_pending_job(
    db_session, seeded_client, worker_sessionmaker, dispatch_registry,
    forbid_execute_job,
) -> None:
    # Cancelled / replayed jobs must not be re-executed.
    job = await _seed_job(
        db_session, seeded_client[0].id, task_type=_task_type("t"),
        status=JobStatus.COMPLETE.value,
    )
    await runner._dispatch_job(job.id)
    await db_session.refresh(job)
    assert job.status == JobStatus.COMPLETE.value


# --- tts branch ---


async def test_dispatch_tts_uses_tts_executor(
    db_session, seeded_client, worker_sessionmaker, dispatch_registry,
    forbid_execute_job, monkeypatch,
) -> None:
    import tts.executor as tts_executor

    calls = {}

    async def fake_tts(*, job, client, session):
        calls["job_id"] = job.id
        calls["client_id"] = client.id

    monkeypatch.setattr(tts_executor, "execute_tts", fake_tts)
    job = await _seed_job(db_session, seeded_client[0].id, task_type="tts")
    await runner._dispatch_job(job.id)
    assert calls["job_id"] == job.id
    assert calls["client_id"] == seeded_client[0].id


# --- media branch ---


async def test_dispatch_media_rule_routes_to_media_executor(
    db_session, seeded_client, worker_sessionmaker, dispatch_registry,
    forbid_execute_job, monkeypatch,
) -> None:
    import worker.executor as executor

    captured = {}

    async def fake_media(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(executor, "execute_media_job", fake_media)
    task_type = _task_type("img")
    await _seed_media_rule(
        db_session, task_type=task_type, media_kind="image",
        preferred_model="wander_scene_image",
    )
    job = await _seed_job(db_session, seeded_client[0].id, task_type=task_type)
    await runner._dispatch_job(job.id)

    assert captured["job"].id == job.id
    assert captured["media_provider_name"] == "comfyui"
    assert captured["media_kind"] == "image"
    assert captured["workflow_template"] == "wander_scene_image"
    assert captured["extra_params"] == {}
    assert captured["providers"] is dispatch_registry


async def test_dispatch_media_style_override_beats_rule_template(
    db_session, seeded_client, worker_sessionmaker, dispatch_registry,
    forbid_execute_job, monkeypatch,
) -> None:
    # /image stamps style_resolved on the job at submit (#53); a job-level
    # template must beat the rule default and its params must flow through.
    import worker.executor as executor

    captured = {}

    async def fake_media(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(executor, "execute_media_job", fake_media)
    task_type = _task_type("img")
    await _seed_media_rule(
        db_session, task_type=task_type, media_kind="image",
        preferred_model="wander_scene_image",
    )
    job = await _seed_job(
        db_session, seeded_client[0].id, task_type=task_type,
        job_metadata={"style_resolved": {
            "workflow_template": "ghibli_v2", "params": {"width": 512},
        }},
    )
    await runner._dispatch_job(job.id)

    assert captured["workflow_template"] == "ghibli_v2"
    assert captured["extra_params"] == {"width": 512}


# --- code_eval / dpo_fine_tune branches ---


async def test_dispatch_code_eval_uses_code_eval_executor(
    db_session, seeded_client, worker_sessionmaker, dispatch_registry,
    forbid_execute_job, monkeypatch,
) -> None:
    calls = {}

    async def fake_code_eval(*, job, client, session):
        calls["job_id"] = job.id

    monkeypatch.setattr(runner, "execute_code_eval_job", fake_code_eval)
    job = await _seed_job(db_session, seeded_client[0].id, task_type="code_eval")
    await runner._dispatch_job(job.id)
    assert calls["job_id"] == job.id


async def test_dispatch_dpo_fine_tune_gets_providers(
    db_session, seeded_client, worker_sessionmaker, dispatch_registry,
    forbid_execute_job, monkeypatch,
) -> None:
    calls = {}

    async def fake_dpo(*, job, client, session, providers):
        calls["job_id"] = job.id
        calls["providers"] = providers

    monkeypatch.setattr(runner, "execute_dpo_fine_tune_job", fake_dpo)
    job = await _seed_job(db_session, seeded_client[0].id, task_type="dpo_fine_tune")
    await runner._dispatch_job(job.id)
    assert calls["job_id"] == job.id
    # The dpo executor frees/re-pins residents, so it needs the live registry.
    assert calls["providers"] is dispatch_registry


# --- routing / sensitivity ---


async def test_dispatch_sensitivity_violation_fails_job(
    db_session, seeded_client, worker_sessionmaker, dispatch_registry,
    forbid_execute_job,
) -> None:
    # A confidential job explicitly requesting a cloud model must never
    # execute — the routing engine raises and the worker records the failure.
    job = await _seed_job(
        db_session, seeded_client[0].id, task_type=_task_type("t"),
        sensitivity="confidential", model_requested="claude-sonnet-4-20250514",
    )
    await runner._dispatch_job(job.id)
    await db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value
    assert job.error.startswith("routing:")
    assert "disallowed" in job.error


# --- judge branch ---


async def test_dispatch_judge_job_uses_judge_executor(
    db_session, seeded_client, worker_sessionmaker, dispatch_registry,
    forbid_execute_job, monkeypatch,
) -> None:
    calls = {}

    async def fake_judge(*, job, decision, client, providers, session, rule):
        calls["job_id"] = job.id
        calls["decision"] = decision
        calls["rule"] = rule

    monkeypatch.setattr(runner, "execute_judge_job", fake_judge)
    # target_job_id in inputs is the judge marker (is_judge_job).
    job = await _seed_job(
        db_session, seeded_client[0].id, task_type=_task_type("judge"),
        inputs={"target_job_id": str(uuid4())},
    )
    await runner._dispatch_job(job.id)
    assert calls["job_id"] == job.id
    # No rule exists for this task_type, so routing fell back to the default
    # model — which the stub ollama reports as already loaded (no swap).
    assert calls["decision"].model == "llama3.3:70b"
    assert calls["rule"] is None


# --- normal execute + shadow fan-out ---


async def test_dispatch_executes_and_fans_out_shadows(
    db_session, seeded_client, worker_sessionmaker, dispatch_registry,
    monkeypatch,
) -> None:
    import eval.shadow_runner as shadow_runner

    executed = {}
    fanned = {}

    async def fake_execute(*, job, decision, client, providers, session):
        executed["job_id"] = job.id
        executed["model"] = decision.model
        # Real execute_job flips status; should_fan_out_shadows keys on it.
        job.status = JobStatus.COMPLETE.value

    async def fake_enqueue(*, parent_job, rule, client, session):
        fanned["parent_id"] = parent_job.id
        return [uuid4(), uuid4()]

    monkeypatch.setattr(runner, "execute_job", fake_execute)
    monkeypatch.setattr(shadow_runner, "enqueue_shadows_for_parent", fake_enqueue)
    job = await _seed_job(db_session, seeded_client[0].id, task_type=_task_type("t"))
    await runner._dispatch_job(job.id)
    assert executed["job_id"] == job.id
    assert executed["model"] == "llama3.3:70b"
    assert fanned["parent_id"] == job.id


async def test_dispatch_swap_failure_blocks_execution(
    db_session, seeded_client, worker_sessionmaker, dispatch_registry,
    forbid_execute_job, monkeypatch,
) -> None:
    # Target model isn't loaded and the load blows up — the job must fail
    # with per-model attribution and the executor must never run.
    async def boom_load(model, keep_alive=None):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(dispatch_registry.get("ollama"), "load", boom_load)
    task_type = _task_type("t")
    rule = RoutingRule(
        task_type=task_type, preferred_model="qwen3:32b", fallback_model="qwen3:32b",
    )
    db_session.add(rule)
    await db_session.commit()
    job = await _seed_job(db_session, seeded_client[0].id, task_type=task_type)
    await runner._dispatch_job(job.id)
    await db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value
    assert "model swap failed" in job.error
    assert job.model_used == "qwen3:32b"


# --- _swap_ollama_if_needed (direct) ---


class _Span:
    def set_attribute(self, *_args) -> None:
        pass

    def record_exception(self, *_args) -> None:
        pass


def _decision(provider="ollama", model="qwen3:32b"):
    return SimpleNamespace(provider=provider, model=model)


async def test_swap_noop_for_non_ollama_provider(stub_registry) -> None:
    ok = await runner._swap_ollama_if_needed(
        _decision(provider="anthropic", model="claude-sonnet-4-20250514"),
        stub_registry, None, "c", None, _Span(),
    )
    assert ok is True


async def test_swap_skipped_when_model_already_loaded(stub_registry) -> None:
    loads = []

    async def record_load(model, keep_alive=None):
        loads.append(model)

    stub_registry.get("ollama").load = record_load
    ok = await runner._swap_ollama_if_needed(
        _decision(model="llama3.3:70b"), stub_registry, None, "c", None, _Span(),
    )
    assert ok is True
    assert loads == []  # already resident — no swap traffic


async def test_swap_loads_target_and_records_duration(
    db_session, seeded_client, stub_registry
) -> None:
    loads = []

    async def record_load(model, keep_alive=None):
        loads.append(model)

    stub_registry.get("ollama").load = record_load
    job = await _seed_job(db_session, seeded_client[0].id, task_type=_task_type("t"))
    ok = await runner._swap_ollama_if_needed(
        _decision(model="qwen3:32b"), stub_registry, job, "c", db_session, _Span(),
    )
    assert ok is True
    assert loads == ["qwen3:32b"]
    assert job.job_metadata["model_swap_ms"] >= 0


async def test_swap_tolerates_list_loaded_failure(stub_registry) -> None:
    # /api/ps failing must not block the swap — treat as nothing loaded.
    async def boom_ps():
        raise RuntimeError("ps failed")

    loads = []

    async def record_load(model, keep_alive=None):
        loads.append(model)

    ollama = stub_registry.get("ollama")
    ollama.list_loaded = boom_ps
    ollama.load = record_load
    job = Job(
        client_app_id=uuid4(), task_type="t", sensitivity="internal", prompt="",
        job_metadata={},
    )
    ok = await runner._swap_ollama_if_needed(
        _decision(model="qwen3:32b"), stub_registry, job, "c", _NoCommit(), _Span(),
    )
    assert ok is True
    assert loads == ["qwen3:32b"]


class _NoCommit:
    async def commit(self) -> None:
        pass


# --- _media_dispatch_for_rule / _apply_style_override ---


@pytest.mark.parametrize(
    ("media_kind", "expected"),
    [
        ("image", ("comfyui", "tpl", {})),
        ("video", ("comfyui", "tpl", {})),
        ("audio", ("acestep", "", {})),
        ("mux", ("ffmpeg_mux", "", {})),
    ],
)
def test_media_dispatch_for_rule(media_kind, expected) -> None:
    rule = RoutingRule(
        task_type="t", preferred_model="tpl", fallback_model="",
        media_kind=media_kind,
    )
    assert runner._media_dispatch_for_rule(rule) == expected


def test_media_dispatch_unknown_kind_raises() -> None:
    rule = RoutingRule(
        task_type="t", preferred_model="tpl", fallback_model="", media_kind="hologram",
    )
    with pytest.raises(ValueError, match="unknown media_kind"):
        runner._media_dispatch_for_rule(rule)


def test_apply_style_override_passthrough_without_style() -> None:
    job = Job(
        client_app_id=uuid4(), task_type="t", sensitivity="public", prompt="",
        job_metadata=None,
    )
    assert runner._apply_style_override(job, "rule_tpl", {"fps": 24}) == (
        "rule_tpl", {"fps": 24},
    )


def test_apply_style_override_template_and_param_merge() -> None:
    job = Job(
        client_app_id=uuid4(), task_type="t", sensitivity="public", prompt="",
        job_metadata={"style_resolved": {
            "workflow_template": "ghibli", "params": {"fps": 30, "seed": 7},
        }},
    )
    tpl, params = runner._apply_style_override(job, "rule_tpl", {"fps": 24, "width": 512})
    assert tpl == "ghibli"
    # Style params win on collision, rule params survive otherwise.
    assert params == {"fps": 30, "width": 512, "seed": 7}


# --- _get_providers registry construction ---


@pytest.fixture
def reset_providers(monkeypatch):
    monkeypatch.setattr(runner, "_providers", None)


def _fake_settings(anthropic_api_key=None):
    return SimpleNamespace(
        ollama_base_url="http://localhost:11434",
        ollama_num_ctx=4096,
        anthropic_api_key=anthropic_api_key,
    )


def test_get_providers_local_only(reset_providers, monkeypatch) -> None:
    monkeypatch.setattr(runner, "get_settings", _fake_settings)
    reg = runner._get_providers()
    assert reg.has("ollama")
    assert not reg.has("anthropic")  # no key configured
    # Media providers registered with the same defaults as the API lifespan.
    assert reg.has_media("comfyui")
    assert reg.has_media("acestep")
    assert reg.has_media("ffmpeg_mux")
    # Process-global: second call returns the cached registry.
    assert runner._get_providers() is reg


def test_get_providers_registers_anthropic_when_keyed(
    reset_providers, monkeypatch
) -> None:
    monkeypatch.setattr(
        runner, "get_settings", lambda: _fake_settings(anthropic_api_key="sk-test")
    )
    reg = runner._get_providers()
    assert reg.has("anthropic")


def test_get_providers_media_failure_is_nonfatal(reset_providers, monkeypatch) -> None:
    # A dead media daemon (constructor raising) must not take out text serving.
    import providers.comfyui as comfyui

    class _Boom:
        def __init__(self, **_kwargs) -> None:
            raise RuntimeError("comfy not reachable")

    monkeypatch.setattr(runner, "get_settings", _fake_settings)
    monkeypatch.setattr(comfyui, "ComfyUIProvider", _Boom)
    reg = runner._get_providers()
    assert reg.has("ollama")
    assert not reg.has_media("comfyui")
    assert reg.has_media("acestep")  # the others still registered


def test_get_providers_all_media_down_still_serves_text(
    reset_providers, monkeypatch
) -> None:
    import providers.acestep as acestep
    import providers.comfyui as comfyui
    import providers.ffmpeg_mux as ffmpeg_mux

    class _Boom:
        def __init__(self, **_kwargs) -> None:
            raise RuntimeError("daemon down")

    monkeypatch.setattr(runner, "get_settings", _fake_settings)
    monkeypatch.setattr(comfyui, "ComfyUIProvider", _Boom)
    monkeypatch.setattr(acestep, "ACEStepProvider", _Boom)
    monkeypatch.setattr(ffmpeg_mux, "FFmpegMuxProvider", _Boom)
    reg = runner._get_providers()
    assert reg.has("ollama")
    for name in ("comfyui", "acestep", "ffmpeg_mux"):
        assert not reg.has_media(name)


# --- _mark_job_failed_safe ---


async def test_mark_failed_safe_swallows_session_failure(monkeypatch, caplog) -> None:
    # The safe marker runs after dispatch already blew up — if even its own
    # fresh session can't be built, it must log and return, never raise.
    def broken_maker():
        raise RuntimeError("db gone")

    monkeypatch.setattr(runner, "get_worker_session_maker", broken_maker)
    await runner._mark_job_failed_safe(uuid4(), "original error")
    assert any("failed to mark job" in r.message for r in caplog.records)
