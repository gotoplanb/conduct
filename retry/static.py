"""Static failure handling — v1 implementation.

Decision rule: if the routing decision has a fallback model AND its provider is
loaded in the registry, return FALLBACK. Otherwise FAIL. This mirrors what the
executor used to do inline; pulling it behind the FailureHandler interface is
the seam that v2's triage model plugs into.
"""

from __future__ import annotations

from retry.base import FailureContext, FailureHandler, HandlerAction, HandlerDecision


class StaticFailureHandler(FailureHandler):
    async def on_provider_error(self, ctx: FailureContext) -> HandlerDecision:
        decision = ctx.decision
        if (
            decision.fallback_model
            and decision.fallback_provider
            and decision.fallback_provider in ctx.available_providers
            and not (
                decision.fallback_model == decision.model
                and decision.fallback_provider == decision.provider
            )
        ):
            return HandlerDecision(
                action=HandlerAction.FALLBACK,
                target_model=decision.fallback_model,
                target_provider=decision.fallback_provider,
                reason=ctx.error_type,
            )
        return HandlerDecision(action=HandlerAction.FAIL, reason=ctx.error_type)
