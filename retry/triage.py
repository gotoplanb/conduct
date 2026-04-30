"""Triage failure handler — v2 stub.

When wired up, this consults a small always-resident local model (3-8B) with
operational metadata (no prompt content) and returns a HandlerDecision. The
prompt lives at `prompts/shared/triage.md` and is already authored.

Wiring path when ready:
  1. Build a TriageFailureHandler(provider=triage_model_name) in lifespan
  2. Set app.state.failure_handler = TriageFailureHandler(...)
  3. Same deal in worker/runner.py for the worker process

The interface is identical to StaticFailureHandler so swapping is mechanical.
"""

from __future__ import annotations

from retry.base import FailureContext, FailureHandler, HandlerDecision


class TriageFailureHandler(FailureHandler):
    def __init__(self, *, model: str = "llama3.2:3b") -> None:
        self.model = model

    async def on_provider_error(self, ctx: FailureContext) -> HandlerDecision:
        raise NotImplementedError(
            "TriageFailureHandler is v2 — use StaticFailureHandler until the "
            "triage model is wired in"
        )
