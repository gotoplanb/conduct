from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from auth import admin_only
from config.pricing import get_pricing
from deps import get_provider_registry
from providers.base import ProviderError
from providers.ollama import OllamaProvider
from providers.registry import ProviderRegistry
from providers.resident import is_resident

log = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"], dependencies=[Depends(admin_only)])


class LocalModel(BaseModel):
    name: str
    status: str  # "loaded" | "unloaded"
    resident: bool = False
    size_gb: float | None = None
    last_used: datetime | None = None


class CloudModel(BaseModel):
    name: str
    provider: str


class ModelsOut(BaseModel):
    local: list[LocalModel]
    cloud: list[CloudModel]


def _ollama(providers: ProviderRegistry) -> OllamaProvider:
    if not providers.has("ollama"):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "ollama provider not configured")
    return providers.get("ollama")  # type: ignore[return-value]


@router.get("")
async def list_models(
    providers: ProviderRegistry = Depends(get_provider_registry),
) -> ModelsOut:
    ollama = _ollama(providers)

    # Local: cross-reference /api/tags (installed) with /api/ps (resident).
    try:
        all_local = await ollama.list_models()
        loaded_local = await ollama.list_loaded()
    except Exception as e:
        log.warning("failed to query Ollama: %s", e)
        all_local = []
        loaded_local = []

    loaded_names = {m.get("name") for m in loaded_local}
    local: list[LocalModel] = []
    for m in all_local:
        size_bytes = m.get("size") or 0
        size_gb = round(size_bytes / (1024**3), 2) if size_bytes else None
        modified = m.get("modified_at")
        last_used: datetime | None = None
        if isinstance(modified, str):
            try:
                last_used = datetime.fromisoformat(modified.replace("Z", "+00:00"))
            except ValueError:
                last_used = None
        local.append(
            LocalModel(
                name=m["name"],
                status="loaded" if m.get("name") in loaded_names else "unloaded",
                resident=is_resident(m["name"]),
                size_gb=size_gb,
                last_used=last_used,
            )
        )

    cloud = [CloudModel(provider=p, name=m) for (p, m) in get_pricing().configured_models()]
    return ModelsOut(local=local, cloud=cloud)


@router.post("/{name:path}/load")
async def load_model(
    name: str,
    providers: ProviderRegistry = Depends(get_provider_registry),
) -> dict:
    ollama = _ollama(providers)
    try:
        await ollama.load(name)
    except ProviderError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"load failed: {e}") from e
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"load failed: {e!r}") from e
    return {"name": name, "status": "loaded"}


@router.post("/{name:path}/unload")
async def unload_model(
    name: str,
    providers: ProviderRegistry = Depends(get_provider_registry),
) -> dict:
    if is_resident(name):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{name} is a resident model — remove it from RESIDENT_MODELS first",
        )
    ollama = _ollama(providers)
    try:
        await ollama.unload(name)
    except ProviderError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"unload failed: {e}") from e
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"unload failed: {e!r}") from e
    return {"name": name, "status": "unloaded"}
