"""Resident-model reconcile + residency suspension (conduct#47).

The worker pins residents at boot, but an Ollama restart evicts them with no
recovery. `reconcile_resident_models` re-pins the gaps; the suspend/resume flag
keeps it from fighting the dpo_fine_tune memory dance.
"""

from __future__ import annotations

import providers.resident as resident
from providers.resident import (
    PIN_FOREVER,
    reconcile_resident_models,
    residency_suspended,
    resume_residency,
    suspend_residency,
)


class _StubOllama:
    """Records load() calls and serves a configurable /api/ps result."""

    name = "ollama"

    def __init__(self, loaded: list[str], *, list_raises: bool = False) -> None:
        self._loaded = loaded
        self._list_raises = list_raises
        self.loaded_calls: list[tuple[str, int | None]] = []

    async def list_loaded(self) -> list[dict]:
        if self._list_raises:
            raise RuntimeError("ollama down")
        return [{"name": n} for n in self._loaded]

    async def load(self, model, keep_alive=None) -> None:
        self.loaded_calls.append((model, keep_alive))


def _set_residents(monkeypatch, names: list[str]) -> None:
    monkeypatch.setattr(resident, "resident_model_names", lambda: names)


async def test_reconcile_repins_only_evicted(monkeypatch) -> None:
    _set_residents(monkeypatch, ["a:1", "b:2", "c:3"])
    # only b:2 is still loaded -> a:1 and c:3 must be re-pinned (keep_alive=-1)
    ollama = _StubOllama(loaded=["b:2"])
    resume_residency()

    repinned = await reconcile_resident_models(ollama)

    assert repinned == ["a:1", "c:3"]
    assert ollama.loaded_calls == [("a:1", PIN_FOREVER), ("c:3", PIN_FOREVER)]


async def test_reconcile_noop_when_all_loaded(monkeypatch) -> None:
    _set_residents(monkeypatch, ["a:1", "b:2"])
    ollama = _StubOllama(loaded=["a:1", "b:2"])
    resume_residency()

    repinned = await reconcile_resident_models(ollama)

    assert repinned == []
    assert ollama.loaded_calls == []


async def test_reconcile_skipped_while_suspended(monkeypatch) -> None:
    # During a dpo_fine_tune job the set is intentionally unloaded — reconcile
    # must not re-pin and re-contend GPU memory.
    _set_residents(monkeypatch, ["a:1", "b:2"])
    ollama = _StubOllama(loaded=[])  # everything evicted
    suspend_residency()
    try:
        repinned = await reconcile_resident_models(ollama)
    finally:
        resume_residency()

    assert repinned == []
    assert ollama.loaded_calls == []


async def test_reconcile_best_effort_when_ollama_down(monkeypatch) -> None:
    _set_residents(monkeypatch, ["a:1"])
    ollama = _StubOllama(loaded=[], list_raises=True)
    resume_residency()

    repinned = await reconcile_resident_models(ollama)  # must not raise

    assert repinned == []
    assert ollama.loaded_calls == []


async def test_reconcile_empty_resident_set(monkeypatch) -> None:
    _set_residents(monkeypatch, [])
    ollama = _StubOllama(loaded=[])
    resume_residency()

    assert await reconcile_resident_models(ollama) == []


def test_suspend_resume_toggle() -> None:
    resume_residency()
    assert residency_suspended() is False
    suspend_residency()
    assert residency_suspended() is True
    resume_residency()
    assert residency_suspended() is False
