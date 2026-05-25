"""FastAPI lifespan: provider registry, OTel tracing, auto-instrumentation, SIGHUP."""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
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
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()

    # Providers.
    registry = ProviderRegistry()
    registry.register(OllamaProvider(base_url=settings.ollama_base_url))
    if settings.anthropic_api_key:
        registry.register(AnthropicProvider(api_key=settings.anthropic_api_key))
    else:
        log.warning("ANTHROPIC_API_KEY not set — Anthropic-routed jobs will fail.")
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
