"""Integration tests for the dpo_fine_tune executor (worker.executor, #45).

dpo_fine_tune is a thin provider: it pulls the calling client's own preference
pairs and POSTs them to the external MLX training sidecar. The sidecar is
stubbed here (Conduct carries no ML stack); the real mlx-tune training is
validated against the live sidecar separately.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from codegen.train_client import TrainResult, TrainServiceError
from models.job import Job
from models.types import JobStatus
from worker.executor import execute_dpo_fine_tune_job

_QS_DIMS = {"compile": 5, "golden": 5}
_QS_LOW = {"compile": 1, "golden": 1}


class _FakeTrain:
    def __init__(self, result: TrainResult | None = None, *, raise_error: bool = False) -> None:
        self._result = result
        self._raise = raise_error
        self.seen: dict | None = None

    async def train(self, *, base_model, pairs, training=None) -> TrainResult:
        if self._raise:
            raise TrainServiceError("training sidecar unreachable: boom")
        self.seen = {"base_model": base_model, "pairs": pairs, "training": training}
        return self._result


async def _job(db, client_id, *, response, dims, model="m") -> Job:
    job = Job(
        client_app_id=client_id, task_type="code_generation", sensitivity="internal",
        prompt="P?", response=response, model_used=model, status=JobStatus.COMPLETE.value,
        job_metadata={"quality_scores": [{"score": round(sum(dims.values()) / len(dims)),
                                          "scores": dims, "via": "code-eval"}]},
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _seed_pair(db, client_id) -> None:
    """A parent + shadow on the same prompt with a composite gap -> one pair."""
    parent = await _job(db, client_id, response="good", dims=_QS_DIMS, model="big")
    from models.shadow import JobShadow
    sh = JobShadow(
        parent_job_id=parent.id, model="small", provider="ollama", response="bad",
        status=JobStatus.COMPLETE.value,
        shadow_metadata={"quality_scores": [{"score": 1, "scores": _QS_LOW, "via": "code-eval"}]},
    )
    db.add(sh)
    await db.commit()


async def _seed_dpo_job(db, client_id, **extra_inputs) -> Job:
    job = Job(
        client_app_id=client_id, task_type="dpo_fine_tune", sensitivity="internal",
        prompt="", status=JobStatus.PENDING.value,
        inputs={"base_model": "gemma4:e4b", "source_task_type": "code_generation",
                "method": "composite", "min_gap": 2, **extra_inputs},
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def test_dpo_fine_tune_trains_and_records_tag(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    await _seed_pair(db_session, client.id)
    job = await _seed_dpo_job(db_session, client.id)
    result = TrainResult(
        tag="gemma4-e4b-dpo-abc123", artifact_path="/m5/out/x.gguf",
        pairs_consumed=1, training_time_s=42.0, dataset_sha="deadbeef",
    )
    stub = _FakeTrain(result)

    out = await execute_dpo_fine_tune_job(
        job=job, client=client, session=db_session, train_client=stub,
    )

    assert out.status == JobStatus.COMPLETE.value
    assert out.model_used == "gemma4-e4b-dpo-abc123"  # the new servable tag
    meta = out.job_metadata["training"]
    assert meta["tag"] == "gemma4-e4b-dpo-abc123"
    assert meta["base_model"] == "gemma4:e4b" and meta["dataset_sha"] == "deadbeef"
    assert meta["pairs_submitted"] == 1
    # the client's own pairs reached the sidecar
    assert stub.seen["base_model"] == "gemma4:e4b"
    assert len(stub.seen["pairs"]) == 1
    assert stub.seen["pairs"][0]["chosen"] == "good"


async def test_dpo_fine_tune_no_pairs_fails(db_session: AsyncSession, seeded_client) -> None:
    client = seeded_client[0]
    # no preference data seeded
    job = await _seed_dpo_job(db_session, client.id)
    out = await execute_dpo_fine_tune_job(
        job=job, client=client, session=db_session, train_client=_FakeTrain(),
    )
    assert out.status == JobStatus.FAILED.value
    assert "no preference pairs" in out.error


async def test_dpo_fine_tune_missing_base_model_fails(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    job = Job(
        client_app_id=client.id, task_type="dpo_fine_tune", sensitivity="internal",
        prompt="", status=JobStatus.PENDING.value, inputs={"source_task_type": "code_generation"},
    )
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)
    out = await execute_dpo_fine_tune_job(
        job=job, client=client, session=db_session, train_client=_FakeTrain(),
    )
    assert out.status == JobStatus.FAILED.value
    assert "base_model is required" in out.error


async def test_dpo_fine_tune_sidecar_error_fails_loudly(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    await _seed_pair(db_session, client.id)
    job = await _seed_dpo_job(db_session, client.id)
    out = await execute_dpo_fine_tune_job(
        job=job, client=client, session=db_session, train_client=_FakeTrain(raise_error=True),
    )
    assert out.status == JobStatus.FAILED.value
    assert "dpo_fine_tune:" in out.error and "unreachable" in out.error


class _StubOllama:
    name = "ollama"

    def __init__(self) -> None:
        self.unloaded: list[str] = []
        self.pinned: list[str] = []

    async def unload(self, model) -> None:
        self.unloaded.append(model)

    async def load(self, model, keep_alive=None) -> None:
        self.pinned.append(model)


class _StubRegistry:
    def __init__(self, ollama) -> None:
        self._ollama = ollama

    def has(self, name) -> bool:
        return name == "ollama"

    def get(self, name):
        return self._ollama


async def test_dpo_fine_tune_frees_and_repins_resident_models(
    db_session: AsyncSession, seeded_client, monkeypatch
) -> None:
    # Conduct owns the GPU-memory dance: unload resident models before training,
    # re-pin after (#45). Verify both happen around a successful train.
    import providers.resident as resident
    monkeypatch.setattr(resident, "resident_model_names", lambda: ["gemma4:e4b", "llama3.2:3b"])
    client = seeded_client[0]
    await _seed_pair(db_session, client.id)
    job = await _seed_dpo_job(db_session, client.id)
    ollama = _StubOllama()
    result = TrainResult(
        tag="t", artifact_path="/x", pairs_consumed=1, training_time_s=1.0, dataset_sha="s",
    )

    out = await execute_dpo_fine_tune_job(
        job=job, client=client, session=db_session,
        providers=_StubRegistry(ollama), train_client=_FakeTrain(result),
    )

    assert out.status == JobStatus.COMPLETE.value
    assert ollama.unloaded == ["gemma4:e4b", "llama3.2:3b"]  # freed before train
    assert ollama.pinned == ["gemma4:e4b", "llama3.2:3b"]    # re-pinned after


async def test_dpo_fine_tune_repins_even_on_failure(
    db_session: AsyncSession, seeded_client, monkeypatch
) -> None:
    # Serving must be restored even if training fails.
    import providers.resident as resident
    monkeypatch.setattr(resident, "resident_model_names", lambda: ["gemma4:e4b"])
    client = seeded_client[0]
    await _seed_pair(db_session, client.id)
    job = await _seed_dpo_job(db_session, client.id)
    ollama = _StubOllama()

    out = await execute_dpo_fine_tune_job(
        job=job, client=client, session=db_session,
        providers=_StubRegistry(ollama), train_client=_FakeTrain(raise_error=True),
    )

    assert out.status == JobStatus.FAILED.value
    assert ollama.unloaded == ["gemma4:e4b"]
    assert ollama.pinned == ["gemma4:e4b"]  # re-pinned despite the failure


async def test_dpo_fine_tune_scopes_to_caller(db_session: AsyncSession, seeded_client) -> None:
    """Pairs from ANOTHER client must not be trained on."""
    from uuid import uuid4

    from models.client import ClientApp
    mine = seeded_client[0]
    other = ClientApp(name=f"other-{uuid4().hex[:6]}", api_key_hash=uuid4().hex)
    db_session.add(other)
    await db_session.commit()
    await db_session.refresh(other)
    await _seed_pair(db_session, other.id)  # only the OTHER client has data

    job = await _seed_dpo_job(db_session, mine.id)
    out = await execute_dpo_fine_tune_job(
        job=job, client=mine, session=db_session, train_client=_FakeTrain(),
    )
    assert out.status == JobStatus.FAILED.value
    assert "no preference pairs" in out.error  # mine has none; other's are invisible
