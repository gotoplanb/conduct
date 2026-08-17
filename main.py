from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from lifespan import lifespan
from mcp_server import build_mcp_app
from routes import (
    clients,
    connectors,
    datasets,
    health,
    jobs,
    metrics_json,
    metrics_prom,
    oauth,
    prompts,
    tts,
    ui,
)
from routes import eval as eval_route
from routes import image as image_route
from routes import models as models_route
from routes import routing as routing_route
from routes import voices as voices_route

app = FastAPI(
    title="Conduct",
    description="LLM dispatch service",
    version="0.1.0",
    lifespan=lifespan,
)

# Static assets — favicons, mascot image, anything else the UI references.
_static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

app.include_router(health.router)
app.include_router(clients.router)
app.include_router(connectors.router)
app.include_router(jobs.router)
app.include_router(models_route.router)
app.include_router(routing_route.router)
app.include_router(metrics_prom.router)
app.include_router(metrics_json.router)
app.include_router(eval_route.router)
app.include_router(datasets.router)
app.include_router(prompts.router)
app.include_router(oauth.router)
app.include_router(tts.tts_router)
app.include_router(tts.output_router)
app.include_router(voices_route.router)
app.include_router(voices_route.admin_router)
app.include_router(image_route.image_router)
app.include_router(image_route.styles_router)
app.include_router(image_route.styles_admin_router)
app.include_router(ui.router)

# Remote MCP server for Claude custom connectors. Streamable-HTTP transport
# behind the OAuth bearer gate; its session manager is started in lifespan.
app.mount("/mcp", build_mcp_app())


class MCPTrailingSlashRewrite:
    """Treat /mcp as /mcp/ so MCP clients aren't bounced through a 307.

    Claude's connector POSTs to /mcp; Starlette's Mount would otherwise
    redirect, doubling every MCP call into two HTTP requests.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/mcp":
            scope["path"] = "/mcp/"
            scope["raw_path"] = b"/mcp/"
        await self.app(scope, receive, send)


app.add_middleware(MCPTrailingSlashRewrite)

# Must run at import time, not in the lifespan: Starlette freezes the
# middleware stack before the lifespan body executes, so instrumenting there
# leaves the OTel middleware out of the request path entirely — the API logs
# "tracing initialized" and then emits zero server spans (#49). The tracer
# provider itself is still set later by init_tracing() in the lifespan; the
# proxy tracer picks it up.
FastAPIInstrumentor.instrument_app(app)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/jobs")
