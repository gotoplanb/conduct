"""Unit tests for the media-provider surface: BaseMediaProvider abstract
interface, MediaResponse shape, and ProviderRegistry's media namespace."""

from __future__ import annotations

from decimal import Decimal

import pytest

from providers.media_base import BaseMediaProvider, MediaResponse
from providers.registry import ProviderRegistry


class _StubMediaProvider(BaseMediaProvider):
    """Bare-minimum impl used in the registry tests below — does no I/O."""

    name = "stub"

    async def produce(self, **kwargs) -> MediaResponse:
        return MediaResponse(
            file_path="/tmp/stub.png",
            url_path="/output/stub.png",
            mime_type="image/png",
            width=1, height=1,
            duration_s=None,
            latency_ms=0,
            cost_usd=Decimal("0"),
            model_used="stub",
            provider=self.name,
            extra={},
        )


def test_media_provider_must_implement_produce() -> None:
    """BaseMediaProvider is abstract — instantiating a no-method subclass
    must blow up at construct time, not silently return None at call time."""

    class _Bad(BaseMediaProvider):
        name = "bad"
        # deliberately no produce()

    with pytest.raises(TypeError, match="abstract"):
        _Bad()


def test_media_response_is_frozen() -> None:
    """MediaResponse is a frozen dataclass — once the worker writes one onto
    a Job row, downstream code (rollups, tracing, eval) should not mutate
    it. The frozen guarantee makes that misuse a runtime error."""
    r = MediaResponse(
        file_path="/x", url_path="/y", mime_type="image/png",
        width=1024, height=768, duration_s=None,
        latency_ms=42, cost_usd=Decimal("0.01"),
        model_used="m", provider="p", extra={},
    )
    # Frozen dataclass mutation raises FrozenInstanceError (subclass of AttributeError).
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        r.file_path = "/elsewhere"  # type: ignore[misc]


def test_registry_media_namespace_is_separate_from_text() -> None:
    """Media providers and text providers share a registry but live in
    different lookup tables — a name collision (e.g. both 'anthropic' and a
    hypothetical 'anthropic' image provider) must not conflate the two."""
    reg = ProviderRegistry()
    media = _StubMediaProvider()
    reg.register_media(media)

    assert reg.has_media("stub") is True
    assert reg.has("stub") is False  # not in the text namespace
    assert reg.get_media("stub") is media

    with pytest.raises(KeyError, match="not registered"):
        reg.get("stub")  # text-side lookup must fail loudly


def test_registry_get_media_unknown_raises() -> None:
    reg = ProviderRegistry()
    with pytest.raises(KeyError, match="media provider not registered"):
        reg.get_media("missing")


@pytest.mark.asyncio
async def test_stub_media_provider_round_trip() -> None:
    """The MediaResponse comes back filled with the fields the worker will
    pull onto the Job row. This is the contract for execute_media_job."""
    p = _StubMediaProvider()
    r = await p.produce(
        prompt="hello",
        inputs={},
        output_dir="/tmp",
        output_basename="abc",
    )
    assert r.provider == "stub"
    assert r.mime_type == "image/png"
    assert r.cost_usd == Decimal("0")
