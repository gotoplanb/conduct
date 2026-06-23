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

from config.settings import get_settings
from providers.ollama import OllamaProvider

log = logging.getLogger(__name__)

# Sentinel value Ollama interprets as "never unload."
PIN_FOREVER = -1


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
