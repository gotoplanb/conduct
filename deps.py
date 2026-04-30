"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Request

from providers.registry import ProviderRegistry


def get_provider_registry(request: Request) -> ProviderRegistry:
    return request.app.state.providers
