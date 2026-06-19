"""Integration test for the code-gen artifact wiring in worker.executor (#23).

A `code_generation` job that opts in via `inputs={"artifact": "cargo"}` has its
response parsed into a Cargo project and stored as a tarball on completion;
malformed output flips the job to FAILED with a structured reason rather than
crashing. Plain text jobs (no opt-in) are untouched.
"""

from __future__ import annotations

import tarfile
from decimal import Decimal
from types import SimpleNamespace

from sqlalchemy.ext.asyncio import AsyncSession

import worker.executor as executor
from models.job import Job
from models.types import JobStatus, Sensitivity
from providers.base import BaseProvider, ProviderResponse
from providers.registry import ProviderRegistry
from routing.engine import RoutingDecision
from worker.executor import execute_job

_CARGO = '[package]\nname = "sol"\nversion = "0.1.0"\nedition = "2021"\n'
_MAIN = "fn main() { println!(\"hi\"); }\n"
_GOOD = f"```toml Cargo.toml\n{_CARGO}```\n```rust src/main.rs\n{_MAIN}```\n"


class _StubProvider(BaseProvider):
    name = "ollama"

    def __init__(self, text: str) -> None:
        self._text = text

    async def complete(
        self, prompt="", model="", system_prompt="", max_tokens=1000,
        temperature=None, seed=None, **kwargs,
    ) -> ProviderResponse:
        return ProviderResponse(
            response=self._text, tokens_in=10, tokens_out=20,
            cost_usd=Decimal("0"), latency_ms=1, model_used=model, provider=self.name,
        )


def _decision() -> RoutingDecision:
    return RoutingDecision(
        model="qwen3.5:4b", provider="ollama", fallback_model=None, fallback_provider=None,
        effective_sensitivity=Sensitivity.INTERNAL, max_tokens=2000, reason="test",
        temperature=0.0, deterministic_seed=True,
    )


async def _seed_codegen_job(db, *, client_id, inputs) -> Job:
    job = Job(
        client_app_id=client_id, task_type="code_generation", sensitivity="internal",
        prompt="Write a Rust hello world.", system_prompt="You write Rust.",
        status=JobStatus.PENDING.value, inputs=inputs,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _run(job, text, session, tmp_path, monkeypatch, client) -> Job:
    monkeypatch.setattr(
        executor, "get_settings", lambda: SimpleNamespace(tts_output_dir=str(tmp_path))
    )
    reg = ProviderRegistry()
    reg.register(_StubProvider(text))
    return await execute_job(
        job=job, decision=_decision(), client=client, providers=reg, session=session
    )


async def test_codegen_stores_cargo_artifact(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    client = seeded_client[0]
    job = await _seed_codegen_job(db_session, client_id=client.id, inputs={"artifact": "cargo"})
    out = await _run(job, _GOOD, db_session, tmp_path, monkeypatch, client)

    assert out.status == JobStatus.COMPLETE.value
    assert out.media_url == f"/output/{job.id}.tar"
    art = (out.job_metadata or {}).get("artifact") or {}
    assert art["format"] == "cargo"
    assert art["files"] == ["Cargo.toml", "src/main.rs"]
    tar_path = tmp_path / f"{job.id}.tar"
    assert tar_path.is_file()
    with tarfile.open(tar_path) as tar:
        assert sorted(tar.getnames()) == ["Cargo.toml", "src/main.rs"]


async def test_codegen_malformed_output_marks_failed_structured(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    client = seeded_client[0]
    job = await _seed_codegen_job(db_session, client_id=client.id, inputs={"artifact": "cargo"})
    # Response with no path-tagged files -> structured failure, no crash.
    prose = "Sure! Here's some prose, no project."
    out = await _run(job, prose, db_session, tmp_path, monkeypatch, client)

    assert out.status == JobStatus.FAILED.value
    assert out.error.startswith("artifact:")
    art = (out.job_metadata or {}).get("artifact") or {}
    assert art["error"] == "no_files"
    assert out.media_url is None


async def test_plain_text_job_without_optin_untouched(
    db_session: AsyncSession, seeded_client, tmp_path, monkeypatch
) -> None:
    client = seeded_client[0]
    # No inputs.artifact -> the artifact step is skipped even for a cargo-shaped
    # response; behaves as a normal text completion.
    job = await _seed_codegen_job(db_session, client_id=client.id, inputs={})
    out = await _run(job, _GOOD, db_session, tmp_path, monkeypatch, client)

    assert out.status == JobStatus.COMPLETE.value
    assert out.media_url is None
    assert "artifact" not in (out.job_metadata or {})
    assert not (tmp_path / f"{job.id}.tar").exists()
