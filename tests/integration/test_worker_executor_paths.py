"""Error-path and fallback coverage for worker.executor.

Complements the per-feature suites (judge / media / code_eval): those pin the
happy paths, these pin the failure lanes — provider fallback, judge input
validation, panel juror loss, code_eval target resolution, and the media
chaining resolver's rejects.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import worker.executor as executor
from codegen.artifact import build_cargo_artifact
from codegen.build_client import BuildReport, CommandResult
from models.client import ClientAppUsage
from models.job import Job
from models.routing import RoutingRule
from models.shadow import JobShadow
from models.types import JobStatus, Sensitivity
from providers.base import ProviderError, ProviderResponse
from providers.media_base import BaseMediaProvider, MediaResponse
from providers.registry import ProviderRegistry
from routing.engine import RoutingDecision
from worker.executor import (
    _aggregate_panel,
    _build_panel,
    _panelist_allowed,
    _parse_judge_verdict,
    _parse_pairwise_verdict,
    _resolve_min_panel_n,
    execute_code_eval_job,
    execute_job,
    execute_judge_job,
    execute_media_job,
)


class _Provider:
    """Text-provider double: canned response(s) or a ProviderError."""

    def __init__(
        self, name: str = "ollama", *, fail: bool = False,
        text: str = "ok", texts: list[str] | None = None,
    ) -> None:
        self.name = name
        self.fail = fail
        self.text = text
        self.texts = list(texts) if texts is not None else None
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> ProviderResponse:
        self.calls.append(kwargs)
        if self.fail:
            raise ProviderError(f"{self.name} down")
        out = self.texts.pop(0) if self.texts is not None else self.text
        return ProviderResponse(
            response=out, tokens_in=7, tokens_out=3, cost_usd=Decimal("0.001"),
            latency_ms=5, model_used=kwargs.get("model", ""), provider=self.name,
        )


def _decision(
    *, model: str = "gemma4:e4b", provider: str = "ollama",
    fallback_model: str | None = None, fallback_provider: str | None = None,
    sensitivity: Sensitivity = Sensitivity.INTERNAL,
    temperature: float | None = 0.0, deterministic_seed: bool = True,
) -> RoutingDecision:
    return RoutingDecision(
        model=model, provider=provider,
        fallback_model=fallback_model, fallback_provider=fallback_provider,
        effective_sensitivity=sensitivity, max_tokens=200, reason="test",
        temperature=temperature, deterministic_seed=deterministic_seed,
    )


async def _seed_job(db, client_id, **kw) -> Job:
    defaults: dict[str, Any] = {
        "task_type": "qa", "sensitivity": "internal", "prompt": "hello",
        "system_prompt": "be terse", "status": JobStatus.PENDING.value,
    }
    defaults.update(kw)
    job = Job(client_app_id=client_id, **defaults)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _run_judge(judge, decision, provider, client, session, rule=None) -> Job:
    reg = ProviderRegistry()
    reg.register(provider)
    return await execute_judge_job(
        job=judge, decision=decision, client=client, providers=reg,
        session=session, rule=rule,
    )


# --- execute_job: fallback lane --------------------------------------------


async def test_primary_failure_falls_back_and_completes(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    job = await _seed_job(db_session, client.id)
    reg = ProviderRegistry()
    primary = _Provider("ollama", fail=True)
    fallback = _Provider("anthropic")
    reg.register(primary)
    reg.register(fallback)

    # temperature=None also exercises the span path that omits the attribute.
    decision = _decision(
        fallback_model="claude-haiku-4-5", fallback_provider="anthropic",
        temperature=None, deterministic_seed=False,
    )
    out = await execute_job(
        job=job, decision=decision, client=client, providers=reg, session=db_session,
    )

    assert out.status == JobStatus.COMPLETE.value
    assert out.model_used == "claude-haiku-4-5"
    assert out.response == "ok"
    assert out.job_metadata["routing"]["used_fallback"] is True
    assert out.job_metadata["prompt"]["source"] == "request_override"
    assert len(primary.calls) == 1
    assert fallback.calls[0]["model"] == "claude-haiku-4-5"
    assert fallback.calls[0]["seed"] is None  # non-deterministic profile


async def test_primary_failure_without_fallback_marks_failed(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    job = await _seed_job(db_session, client.id)
    reg = ProviderRegistry()
    reg.register(_Provider("ollama", fail=True))

    out = await execute_job(
        job=job, decision=_decision(), client=client, providers=reg, session=db_session,
    )

    assert out.status == JobStatus.FAILED.value
    assert out.error == "ProviderError: ollama down"
    assert not out.response
    assert out.job_metadata["routing"]["used_fallback"] is False
    assert out.model_used == "gemma4:e4b"  # failed jobs stay attributable


async def test_fallback_failure_reports_both_errors(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    job = await _seed_job(db_session, client.id)
    reg = ProviderRegistry()
    reg.register(_Provider("ollama", fail=True))
    reg.register(_Provider("anthropic", fail=True))

    decision = _decision(fallback_model="claude-haiku-4-5", fallback_provider="anthropic")
    out = await execute_job(
        job=job, decision=decision, client=client, providers=reg, session=db_session,
    )

    assert out.status == JobStatus.FAILED.value
    assert "primary ProviderError: ollama down" in out.error
    assert "fallback ProviderError: anthropic down" in out.error
    # model_used tracks the fallback attempt for per-model failure analytics.
    assert out.model_used == "claude-haiku-4-5"


async def test_usage_row_accumulates_across_jobs(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    reg = ProviderRegistry()
    reg.register(_Provider("ollama"))
    for _ in range(2):
        job = await _seed_job(db_session, client.id)
        await execute_job(
            job=job, decision=_decision(), client=client, providers=reg, session=db_session,
        )

    usage = await db_session.scalar(
        select(ClientAppUsage).where(ClientAppUsage.client_app_id == client.id)
    )
    assert usage is not None
    assert usage.job_count == 2  # second job updated the existing row
    assert usage.tokens_in == 14
    assert usage.tokens_out == 6


# --- media chaining resolver rejects ----------------------------------------


class _CapturingMedia(BaseMediaProvider):
    name = "capture"

    def __init__(self) -> None:
        self.captured: dict = {}

    async def produce(self, **kwargs):
        self.captured.update(kwargs)
        return MediaResponse(
            file_path="/tmp/x", url_path="/output/x.png", mime_type="image/png",
            width=None, height=None, duration_s=None, latency_ms=0,
            cost_usd=Decimal("0"), model_used="x", provider=self.name, extra={},
        )


async def _run_media(job, provider, session, *, workflow_template="wf") -> Job:
    reg = ProviderRegistry()
    reg.register_media(provider)
    return await execute_media_job(
        job=job, media_provider_name=provider.name, media_kind="image",
        workflow_template=workflow_template, providers=reg,
        output_dir="/tmp/x", session=session,
    )


async def test_media_source_job_id_not_a_uuid_fails(
    db_session: AsyncSession, seeded_client
) -> None:
    job = await _seed_job(
        db_session, seeded_client[0].id, task_type="wander_scene_video",
        inputs={"source_image_job_id": "not-a-uuid"},
    )
    out = await _run_media(job, _CapturingMedia(), db_session)
    assert out.status == JobStatus.FAILED.value
    assert "not a valid UUID" in out.error
    assert "source_image_job_id" in out.error


async def test_media_non_output_url_passed_verbatim(
    db_session: AsyncSession, seeded_client
) -> None:
    # A media_url outside /output/ (e.g. an absolute path or external URL)
    # gets no worker-local translation — it's handed through as-is.
    upstream = await _seed_job(
        db_session, seeded_client[0].id, task_type="wander_scene_image",
        status=JobStatus.COMPLETE.value, media_url="https://cdn.example.com/f.png",
    )
    job = await _seed_job(
        db_session, seeded_client[0].id, task_type="wander_scene_video",
        inputs={"source_image_job_id": str(upstream.id)},
    )
    provider = _CapturingMedia()
    out = await _run_media(job, provider, db_session)
    assert out.status == JobStatus.COMPLETE.value
    assert provider.captured["inputs"]["source_image_url"] == "https://cdn.example.com/f.png"


async def test_media_empty_workflow_template_omits_param(
    db_session: AsyncSession, seeded_client
) -> None:
    job = await _seed_job(db_session, seeded_client[0].id, task_type="wander_scene_image")
    provider = _CapturingMedia()
    out = await _run_media(job, provider, db_session, workflow_template="")
    assert out.status == JobStatus.COMPLETE.value
    assert "workflow_template" not in provider.captured["params"]


# --- verdict parsers --------------------------------------------------------


def test_parse_judge_verdict_dimension_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        _parse_judge_verdict('{"scores": {"humor": 9}}', ["humor"])


def test_parse_pairwise_verdict_requires_json() -> None:
    with pytest.raises(ValueError, match="no JSON"):
        _parse_pairwise_verdict("B wins, obviously")


# --- judge dispatch + validation -------------------------------------------


async def _seed_target(
    db, client_id, *, sensitivity="internal", response="Paris.", model_used=""
) -> Job:
    return await _seed_job(
        db, client_id, task_type="qa", sensitivity=sensitivity,
        prompt="Capital of France?", response=response, model_used=model_used,
        status=JobStatus.COMPLETE.value, system_prompt="",
    )


async def _seed_judge(db, client_id, inputs: dict) -> Job:
    return await _seed_job(
        db, client_id, task_type="judge", prompt="(judge)",
        system_prompt="Score 1-5.", inputs=inputs,
    )


async def test_unknown_judge_mode_fails(db_session: AsyncSession, seeded_client) -> None:
    client = seeded_client[0]
    judge = await _seed_judge(db_session, client.id, {"mode": "vibes", "target_job_id": "x"})
    out = await _run_judge(judge, _decision(), _Provider(), client, db_session)
    assert out.status == JobStatus.FAILED.value
    assert "judge mode 'vibes' not supported" in out.error


async def test_pointwise_invalid_target_uuid_fails(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    judge = await _seed_judge(db_session, client.id, {"target_job_id": "nope"})
    out = await _run_judge(judge, _decision(), _Provider(), client, db_session)
    assert out.status == JobStatus.FAILED.value
    assert "target_job_id='nope' is not a valid UUID" in out.error


async def test_pointwise_provider_error_fails(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id)
    judge = await _seed_judge(db_session, client.id, {"target_job_id": str(target.id)})
    out = await _run_judge(judge, _decision(), _Provider(fail=True), client, db_session)
    assert out.status == JobStatus.FAILED.value
    assert "judge provider error" in out.error


async def test_pointwise_judge_targets_a_shadow(
    db_session: AsyncSession, seeded_client
) -> None:
    # A shadow is judgeable: prompt + sensitivity come from its parent, the
    # response is the shadow's own.
    client = seeded_client[0]
    parent = await _seed_target(db_session, client.id)
    shadow = JobShadow(
        parent_job_id=parent.id, model="alt:1b", provider="ollama",
        status=JobStatus.COMPLETE.value, response="shadow says Paris",
    )
    db_session.add(shadow)
    await db_session.commit()
    await db_session.refresh(shadow)

    judge = await _seed_judge(db_session, client.id, {"target_job_id": str(shadow.id)})
    provider = _Provider(text='{"score": 4, "rationale": "fine"}')
    out = await _run_judge(judge, _decision(), provider, client, db_session)

    assert out.status == JobStatus.COMPLETE.value
    assert out.job_metadata["judge"]["target_kind"] == "shadow"
    user_prompt = provider.calls[0]["prompt"]
    assert parent.prompt in user_prompt
    assert "shadow says Paris" in user_prompt


# --- pairwise validation lanes ---------------------------------------------


async def test_pairwise_identical_targets_fail(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id)
    judge = await _seed_judge(db_session, client.id, {
        "mode": "pairwise", "target_job_id": str(target.id),
        "against_job_id": str(target.id),
    })
    out = await _run_judge(judge, _decision(), _Provider(), client, db_session)
    assert out.status == JobStatus.FAILED.value
    assert "must differ" in out.error


async def test_pairwise_unknown_target_fails(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    against = await _seed_target(db_session, client.id)
    judge = await _seed_judge(db_session, client.id, {
        "mode": "pairwise", "target_job_id": str(uuid4()),
        "against_job_id": str(against.id),
    })
    out = await _run_judge(judge, _decision(), _Provider(), client, db_session)
    assert out.status == JobStatus.FAILED.value
    assert "not found" in out.error


async def test_pairwise_unknown_against_fails(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id)
    judge = await _seed_judge(db_session, client.id, {
        "mode": "pairwise", "target_job_id": str(target.id),
        "against_job_id": str(uuid4()),
    })
    out = await _run_judge(judge, _decision(), _Provider(), client, db_session)
    assert out.status == JobStatus.FAILED.value
    assert "not found" in out.error


async def test_pairwise_provider_error_fails(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    a = await _seed_target(db_session, client.id)
    b = await _seed_target(db_session, client.id, response="London?")
    judge = await _seed_judge(db_session, client.id, {
        "mode": "pairwise", "target_job_id": str(a.id), "against_job_id": str(b.id),
    })
    out = await _run_judge(judge, _decision(), _Provider(fail=True), client, db_session)
    assert out.status == JobStatus.FAILED.value
    assert "judge provider error" in out.error


async def test_pairwise_unparseable_verdict_fails(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    a = await _seed_target(db_session, client.id)
    b = await _seed_target(db_session, client.id, response="London?")
    judge = await _seed_judge(db_session, client.id, {
        "mode": "pairwise", "target_job_id": str(a.id), "against_job_id": str(b.id),
    })
    out = await _run_judge(
        judge, _decision(), _Provider(text="the first one felt stronger"), client, db_session,
    )
    assert out.status == JobStatus.FAILED.value
    assert "could not parse pairwise verdict" in out.error


# --- panel helpers ----------------------------------------------------------


def test_panelist_allowed_cloud_gates() -> None:
    cloud = "claude-haiku-4-5"
    assert _panelist_allowed(cloud, Sensitivity.PUBLIC, False) is True
    assert _panelist_allowed(cloud, Sensitivity.INTERNAL, True) is True
    assert _panelist_allowed(cloud, Sensitivity.INTERNAL, False) is False
    assert _panelist_allowed(cloud, Sensitivity.CONFIDENTIAL, True) is False
    assert _panelist_allowed("gemma4:e4b", Sensitivity.CONFIDENTIAL, False) is True


def test_resolve_min_panel_n_tolerates_garbage() -> None:
    assert _resolve_min_panel_n({"min_panel_n": "lots"}, None) is None
    assert _resolve_min_panel_n({"min_panel_n": 1}, None) is None  # <=1 = no floor


def test_build_panel_dedupes_candidates() -> None:
    eligible, excluded = _build_panel(
        ["llama3.2:3b", "llama3.2:3b", "qwen3.5:4b"],
        "", Sensitivity.INTERNAL, allow_cloud_for_internal=False,
    )
    assert eligible == ["llama3.2:3b", "qwen3.5:4b"]
    assert excluded == {}


def test_aggregate_panel_skips_dimension_nobody_scored() -> None:
    scored = [
        {"model": "a", "score": 4, "scores": {"humor": 4}},
        {"model": "b", "score": 2, "scores": None},
    ]
    agg = _aggregate_panel(scored, ["humor", "clarity"])
    assert agg["dimensions"] == {"humor": 4}  # no clarity median invented


# --- panel dispatch lanes ---------------------------------------------------


async def test_panel_invalid_target_uuid_fails(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    judge = await _seed_judge(db_session, client.id, {"mode": "panel", "target_job_id": "??"})
    out = await _run_judge(judge, _decision(), _Provider(), client, db_session)
    assert out.status == JobStatus.FAILED.value
    assert "is not a valid UUID" in out.error


async def test_panel_unknown_target_fails(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    judge = await _seed_judge(db_session, client.id, {
        "mode": "panel", "target_job_id": str(uuid4()), "panel": ["llama3.2:3b"],
    })
    out = await _run_judge(judge, _decision(), _Provider(), client, db_session)
    assert out.status == JobStatus.FAILED.value
    assert "not found" in out.error


async def test_panel_without_candidates_fails(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id)
    judge = await _seed_judge(db_session, client.id, {
        "mode": "panel", "target_job_id": str(target.id),
    })
    out = await _run_judge(judge, _decision(), _Provider(), client, db_session, rule=None)
    assert out.status == JobStatus.FAILED.value
    assert "panel mode needs panel models" in out.error


async def test_panel_defaults_to_rule_jury(
    db_session: AsyncSession, seeded_client
) -> None:
    # Without inputs.panel the jury is the rule's preferred model plus its
    # eval_shadow_models; malformed shadow entries (no model key) are skipped.
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id, model_used="claude-sonnet-4-6")
    judge = await _seed_judge(db_session, client.id, {
        "mode": "panel", "target_job_id": str(target.id),
    })
    rule = RoutingRule(
        task_type="judge", preferred_model="gemma4:e4b", fallback_model="gemma4:e4b",
        sensitivity="internal",
        eval_shadow_models=[{"model": "llama3.2:3b"}, {"rate": 0.1}],
    )
    provider = _Provider(texts=[
        '{"score": 4, "rationale": "a"}', '{"score": 2, "rationale": "b"}',
    ])
    out = await _run_judge(judge, _decision(), provider, client, db_session, rule=rule)

    assert out.status == JobStatus.COMPLETE.value
    meta = out.job_metadata["judge"]
    assert {p["model"] for p in meta["panelists"]} == {"gemma4:e4b", "llama3.2:3b"}
    assert meta["score"] == 3  # median(4, 2)


async def test_panel_cloud_juror_runs_without_local_load(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id, sensitivity="public")
    judge = await _seed_judge(db_session, client.id, {
        "mode": "panel", "target_job_id": str(target.id), "panel": ["claude-haiku-4-5"],
    })
    provider = _Provider("anthropic", text='{"score": 5, "rationale": "sharp"}')
    out = await _run_judge(
        judge, _decision(sensitivity=Sensitivity.PUBLIC), provider, client, db_session,
    )
    assert out.status == JobStatus.COMPLETE.value
    assert out.job_metadata["judge"]["n"] == 1
    assert provider.calls[0]["model"] == "claude-haiku-4-5"


async def test_panel_all_jurors_erroring_fails(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id, model_used="claude-sonnet-4-6")
    judge = await _seed_judge(db_session, client.id, {
        "mode": "panel", "target_job_id": str(target.id),
        "panel": ["llama3.2:3b", "qwen3.5:4b"],
    })
    out = await _run_judge(judge, _decision(), _Provider(fail=True), client, db_session)
    assert out.status == JobStatus.FAILED.value
    assert "all 2 panelists failed" in out.error
    assert "provider: ollama down" in out.error


# --- code_eval target resolution -------------------------------------------


def _point_output_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        executor, "get_settings",
        lambda: SimpleNamespace(tts_output_dir=str(tmp_path), rust_build_url="http://unused"),
    )


async def _seed_code_eval(db, client_id, inputs: dict) -> Job:
    return await _seed_job(
        db, client_id, task_type="code_eval", prompt="(code_eval)",
        system_prompt="", inputs=inputs,
    )


async def test_code_eval_invalid_target_uuid_fails(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    _point_output_dir(monkeypatch, tmp_path)
    client = seeded_client[0]
    ce = await _seed_code_eval(db_session, client.id, {"target_job_id": "nope"})
    out = await execute_code_eval_job(job=ce, client=client, session=db_session)
    assert out.status == JobStatus.FAILED.value
    assert "invalid target_job_id" in out.error


async def test_code_eval_target_without_artifact_fails(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    _point_output_dir(monkeypatch, tmp_path)
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id)  # plain text job, no artifact
    ce = await _seed_code_eval(db_session, client.id, {"target_job_id": str(target.id)})
    out = await execute_code_eval_job(job=ce, client=client, session=db_session)
    assert out.status == JobStatus.FAILED.value
    assert "has no code artifact" in out.error


async def test_code_eval_missing_artifact_file_fails(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    _point_output_dir(monkeypatch, tmp_path)
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id)
    target.job_metadata = {"artifact": {"url": "/output/vanished.tar"}}
    await db_session.commit()
    ce = await _seed_code_eval(db_session, client.id, {"target_job_id": str(target.id)})
    out = await execute_code_eval_job(job=ce, client=client, session=db_session)
    assert out.status == JobStatus.FAILED.value
    assert "artifact file missing" in out.error


# --- code_eval NA dimensions (mutation / structural / suites) ---------------


def _cmd(command, *, exit_code=0, stdout="", stderr="", timed_out=False) -> CommandResult:
    return CommandResult(command, exit_code, timed_out, 1, stdout, stderr)


class _ScriptedBuild:
    """Returns one canned report per logical command; raises on anything
    unscripted so a test can assert a command never ran."""

    def __init__(self, reports: dict[str, BuildReport]) -> None:
        self._reports = reports

    async def build(self, tar_bytes, commands, **_kw) -> BuildReport:
        assert commands[0] in self._reports, f"unexpected build: {commands}"
        return self._reports[commands[0]]


_TESTS_OK = BuildReport(results={
    "test": _cmd("test", stdout="test result: ok. 2 passed; 0 failed"),
})


async def test_mutation_na_when_not_compiled() -> None:
    score, detail = await executor._run_mutation_dimension(_ScriptedBuild({}), b"", False)
    assert score is None
    assert detail == {"skipped": "did not compile"}


async def test_mutation_na_when_model_tests_fail() -> None:
    bc = _ScriptedBuild({"test": BuildReport(results={
        "test": _cmd("test", exit_code=101, stdout="test result: FAILED. 1 passed; 2 failed"),
    })})
    score, detail = await executor._run_mutation_dimension(bc, b"", True)
    assert score is None
    assert detail["skipped"] == "model tests failing"
    assert detail["passed"] == 1 and detail["failed"] == 2


async def test_mutation_na_when_mutants_time_out() -> None:
    bc = _ScriptedBuild({
        "test": _TESTS_OK,
        "mutants": BuildReport(results={"mutants": _cmd("mutants", timed_out=True)}),
    })
    score, detail = await executor._run_mutation_dimension(bc, b"", True)
    assert score is None
    assert detail["skipped"] == "mutants timed out"


async def test_mutation_na_when_no_viable_mutants() -> None:
    bc = _ScriptedBuild({
        "test": _TESTS_OK,
        "mutants": BuildReport(results={
            "mutants": _cmd("mutants", stdout="3 mutants tested: 3 unviable"),
        }),
    })
    score, detail = await executor._run_mutation_dimension(bc, b"", True)
    assert score is None
    assert detail["skipped"] == "no viable mutants"
    assert detail["unviable"] == 3


async def test_property_suite_failure_without_counterexample() -> None:
    # proptest normally prints a minimized input; when it doesn't, the detail
    # simply omits the key rather than storing an empty one.
    bc = _ScriptedBuild({"test": BuildReport(results={
        "test": _cmd("test", exit_code=101, stdout="test result: FAILED. 0 passed; 2 failed"),
    })})
    suite = {"dimension": "property", "property": True, "files": {"tests/props.rs": "x"}}
    dim, score, detail = await executor._run_suite(bc, b"", suite)
    assert dim == "property"
    assert score == 1  # pass_rate 0
    assert "counterexample" not in detail


_CARGO = '[package]\nname = "x"\nversion = "0.1.0"\nedition = "2021"\n'
_GOOD_PROJECT = (
    f"```toml Cargo.toml\n{_CARGO}```\n"
    "```rust src/lib.rs\npub fn add() -> i64 { 0 }\n```\n"
)


async def test_structural_skips_clippy_when_not_compiled(tmp_path) -> None:
    # tree-sitter still scans non-compiling code, but clippy needs a build —
    # so the clippy leg must not run (the scripted client would raise).
    meta = build_cargo_artifact(_GOOD_PROJECT, job_id=uuid4(), output_dir=tmp_path)
    tar_bytes = (tmp_path / meta["url"].rsplit("/", 1)[-1]).read_bytes()
    score, detail = await executor._run_structural_dimension(
        _ScriptedBuild({}), tar_bytes, {}, False
    )
    assert score == 5  # clean AST alone decides
    assert detail["clippy"] == {"ran": False}
    assert detail["ast"]["ok"] is True
