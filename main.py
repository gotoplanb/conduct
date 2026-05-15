from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from lifespan import lifespan
from routes import clients, health, jobs, metrics_json, metrics_prom, tts, ui
from routes import eval as eval_route
from routes import models as models_route
from routes import routing as routing_route

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
app.include_router(jobs.router)
app.include_router(models_route.router)
app.include_router(routing_route.router)
app.include_router(metrics_prom.router)
app.include_router(metrics_json.router)
app.include_router(eval_route.router)
app.include_router(tts.tts_router)
app.include_router(tts.output_router)
app.include_router(ui.router)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/jobs")
