"""FastAPI lifespan: provider registry, OTel tracing, httpx instrumentation, SIGHUP."""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from config.pricing import get_pricing
from config.settings import get_settings
from observability.logs import init_logging
from observability.tracing import init_tracing
from providers.anthropic import AnthropicProvider
from providers.ollama import OllamaProvider
from providers.registry import ProviderRegistry

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    # Tracing first so subsequent instrumentation captures spans.
    init_tracing(
        service_name=settings.otel_service_name,
        otlp_endpoint=settings.otel_endpoint,
        role="api",
    )
    init_logging(
        service_name=settings.otel_service_name,
        otlp_endpoint=settings.otel_endpoint,
        role="api",
    )
    # FastAPI instrumentation lives in main.py (import time) — attaching it
    # here would miss the middleware stack, which is frozen before the
    # lifespan body runs (#49). httpx patching is library-level, so it's
    # fine (and belongs) here, after init_tracing.
    HTTPXClientInstrumentor().instrument()

    # Providers.
    registry = ProviderRegistry()
    registry.register(OllamaProvider(
        base_url=settings.ollama_base_url, num_ctx=settings.ollama_num_ctx,
    ))
    if settings.anthropic_api_key:
        registry.register(AnthropicProvider(api_key=settings.anthropic_api_key))
    else:
        log.warning("ANTHROPIC_API_KEY not set — Anthropic-routed jobs will fail.")
    _register_media_providers(registry, log)
    app.state.providers = registry
    # Hand the registry to the MCP server so sync-eligible tools run inline.
    from mcp_server import set_provider_registry

    set_provider_registry(registry)

    # Eagerly load pricing so SIGHUP can reload it later.
    pricing = get_pricing()

    def _reload_pricing() -> None:
        try:
            pricing.reload()
            log.warning("pricing reloaded from %s", pricing.path)
        except Exception as e:
            log.exception("pricing reload failed: %s", e)

    # SIGUSR1 (not SIGHUP — uvicorn intercepts SIGHUP for shutdown).
    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGUSR1, _reload_pricing)
        log.warning(
            "SIGUSR1 handler installed for pricing reload "
            "(send via: kill -USR1 $(pgrep -f 'uvicorn main:app'))"
        )
    except (NotImplementedError, RuntimeError) as e:
        # Windows or no running loop in tests
        log.info("SIGUSR1 not available (%s) — pricing must be reloaded by restart", e)

    # Start the MCP server's Streamable-HTTP session manager. The /mcp app is
    # mounted in main.py but its session manager only runs while this context
    # is open, so it lives for the lifetime of the app.
    from mcp_server import mcp

    async with mcp.session_manager.run():
        yield

    HTTPXClientInstrumentor().uninstrument()


def _register_media_providers(registry, log) -> None:
    """Register ComfyUI / ACE-Step / FFmpeg mux providers.

    Reachability: defaults assume the host-side daemons we run as launchd
    services on the M5 Max. Override via env (COMFYUI_BASE_URL,
    ACESTEP_BASE_URL) when deploying elsewhere. Failures here are
    non-fatal — text-only jobs should still work even if a media daemon
    isn't reachable, so we log and continue."""
    import os  # noqa: PLC0415

    try:
        from providers.comfyui import ComfyUIProvider  # noqa: PLC0415

        registry.register_media(
            ComfyUIProvider(
                base_url=os.environ.get(
                    "COMFYUI_BASE_URL", "http://host.docker.internal:8188"
                )
            )
        )
    except Exception:  # noqa: BLE001
        log.warning("ComfyUI media provider unavailable; image/video tasks will fail")

    try:
        from providers.acestep import ACEStepProvider  # noqa: PLC0415

        registry.register_media(
            ACEStepProvider(
                base_url=os.environ.get(
                    "ACESTEP_BASE_URL", "http://host.docker.internal:8002"
                )
            )
        )
    except Exception:  # noqa: BLE001
        log.warning("ACE-Step media provider unavailable; audio tasks will fail")

    try:
        from providers.ffmpeg_mux import FFmpegMuxProvider  # noqa: PLC0415

        registry.register_media(FFmpegMuxProvider())
    except Exception:  # noqa: BLE001
        log.warning("ffmpeg mux provider unavailable; mux tasks will fail")
