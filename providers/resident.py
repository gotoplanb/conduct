"""Resident-model helpers.

A "resident" Ollama model is one the worker pins at boot time (with
`keep_alive=-1`) and promises never to evict. The API may call resident
models directly — bypassing the worker queue — which enables parallel
fan-out for real-time eval across multiple small models.

This module is the single point of truth for which models are resident.
Callers consult `is_resident(model)` rather than re-deriving from settings.
"""

from __future__ import annotations

import logging
import threading

from config.settings import get_settings
from providers.ollama import OllamaProvider

log = logging.getLogger(__name__)

# Sentinel value Ollama interprets as "never unload."
PIN_FOREVER = -1

# When set, residency reconciliation is paused: a dpo_fine_tune job is
# intentionally holding the resident set unloaded to free GPU memory for MLX
# training (see execute_dpo_fine_tune_job). The reconcile loop must NOT re-pin
# during that window — doing so would re-contend the memory the unload exists to
# free and risk the Metal OOM. The DPO executor sets it around train and clears
# it after re-pinning. Cross-thread: the reconcile runs in a daemon thread while
# the DPO job runs in the worker's main thread, so a threading.Event is the
# right primitive.
_residency_suspended = threading.Event()


def suspend_residency() -> None:
    """Pause resident reconciliation (a heavy local job is freeing the GPU)."""
    _residency_suspended.set()


def resume_residency() -> None:
    """Resume resident reconciliation after the resident set is restored."""
    _residency_suspended.clear()


def residency_suspended() -> bool:
    return _residency_suspended.is_set()


def resident_model_names() -> list[str]:
    return get_settings().resident_models


def is_resident(model: str) -> bool:
    return model in resident_model_names()


async def pin_resident_models(ollama: OllamaProvider) -> list[str]:
    """Load every configured resident model with keep_alive=-1.

    Called at worker boot. Failures are logged but do not abort startup —
    a missing model just means residency is unavailable for it; the worker
    can still serve normal traffic.
    """
    pinned: list[str] = []
    for name in resident_model_names():
        try:
            await ollama.load(name, keep_alive=PIN_FOREVER)
            pinned.append(name)
            log.info("pinned resident model: %s", name)
        except Exception as e:
            log.warning("failed to pin resident model %s: %s", name, e)
    return pinned


async def unload_resident_models(ollama: OllamaProvider) -> list[str]:
    """Evict every resident model from GPU memory (keep_alive=0). The inverse of
    pin_resident_models — used to free unified memory for a heavy local job
    (DPO training) on this shared box, since the ~37GB pinned set otherwise
    contends with MLX and OOMs. Best-effort per model; a re-pin restores serving
    afterward. Returns the names unloaded."""
    unloaded: list[str] = []
    for name in resident_model_names():
        try:
            await ollama.unload(name)
            unloaded.append(name)
            log.info("unloaded resident model: %s", name)
        except Exception as e:
            log.warning("failed to unload resident model %s: %s", name, e)
    return unloaded


async def reconcile_resident_models(ollama: OllamaProvider) -> list[str]:
    """Re-pin any resident model that's no longer loaded.

    The worker pins residents once at boot, but the Ollama server can restart
    independently (crash, OOM-abort, KeepAlive respawn, manual restart, binary
    update) and evict the whole set — leaving residents cold with no recovery
    until the worker restarts (conduct#47). A periodic reconcile makes the
    resident guarantee self-healing: it compares the configured set against
    what's actually loaded (`/api/ps`) and re-pins the gaps. Recovers from any
    eviction cause, not just restarts.

    No-op while residency is suspended (a dpo_fine_tune job holds the set down).
    Best-effort and fully logged; never raises. Returns the names re-pinned."""
    if residency_suspended():
        return []
    wanted = resident_model_names()
    if not wanted:
        return []
    try:
        loaded = await ollama.list_loaded()
        loaded_names = {m["name"] for m in loaded}
    except Exception as e:
        log.warning("resident reconcile: could not list loaded models: %s", e)
        return []

    repinned: list[str] = []
    for name in wanted:
        if name in loaded_names:
            continue
        try:
            await ollama.load(name, keep_alive=PIN_FOREVER)
            repinned.append(name)
            log.info("resident reconcile: re-pinned evicted model %s", name)
        except Exception as e:
            log.warning("resident reconcile: failed to re-pin %s: %s", name, e)
    return repinned
