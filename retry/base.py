"""Failure-handling interface shared by the static layer (v1) and the triage
model (v2).

The executor catches a ProviderError, builds a `FailureContext`, and asks the
handler `on_provider_error()` for a `HandlerDecision`. v1 returns "fallback if
available else fail" deterministically; v2 will consult an always-resident
local model that has access to the same context.

The contract is intentionally narrow:
  - The handler does NOT call providers, touch the DB, or modify the Job
  - The handler's only job is to decide what comes next given an error
  - The executor is responsible for honoring the decision (calling fallback,
    marking the job failed, etc.)

This keeps the v2 swap mechanical: change one binding in `lifespan` from
`StaticFailureHandler()` to `TriageFailureHandler(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from routing.engine import RoutingDecision


class HandlerAction(StrEnum):
    FALLBACK = "fallback"  # use decision.fallback_model on its provider
    FAIL = "fail"  # mark job failed, no further attempts
    # Reserved for v2 — static handler never returns these:
    RETRY = "retry"  # retry same model+provider (triage thinks it's transient)
    SWAP = "swap"  # different local model than originally chosen
    RETURN_503 = "return_503"  # tell client to back off
    ESCALATE = "escalate"  # human-attention signal


@dataclass(frozen=True)
class HandlerDecision:
    action: HandlerAction
    target_model: str | None = None
    target_provider: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class FailureContext:
    error_type: str
    error_message: str
    job_task_type: str
    job_sensitivity: str
    decision: RoutingDecision
    available_providers: frozenset[str]


class FailureHandler(Protocol):
    async def on_provider_error(self, ctx: FailureContext) -> HandlerDecision: ...
