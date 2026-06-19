"""Unit tests for codegen.build_client — the rust-build sandbox transport (#24).

Stubs the HTTP layer with httpx.MockTransport so no live build service / cargo
is needed; the real toolchain is validated live against the Watchtower service.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from codegen.build_client import (
    BuildServiceError,
    RustBuildClient,
    compile_summary,
)


def _client(handler) -> RustBuildClient:
    return RustBuildClient("http://build", transport=httpx.MockTransport(handler))


async def test_build_parses_results_and_compiled_flag() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": {
            "check": {"exit": 0, "timed_out": False, "ms": 1200, "stdout": "", "stderr": ""},
        }})

    report = await _client(handler).build(b"tar-bytes", ["check"])
    assert report.compiled is True
    assert report.get("check").ok is True
    assert report.get("check").ms == 1200


async def test_build_sends_base64_tar_and_commands() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"results": {}})

    await _client(handler).build(b"hello-tar", ["check", "build"], timeout_s=42)
    assert base64.b64decode(seen["project_tar_b64"]) == b"hello-tar"
    assert seen["commands"] == ["check", "build"]
    assert seen["timeout_s"] == 42


async def test_compile_summary_failure_surfaces_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # Realistic cargo --message-format=short: path-prefixed diagnostics.
        return httpx.Response(200, json={"results": {"check": {
            "exit": 101, "timed_out": False, "ms": 800,
            "stdout": "",
            "stderr": "src/lib.rs:2:21: error[E0308]: mismatched types\n"
                      "src/lib.rs:5: warning: unused variable: `x`\n",
        }}})

    report = await _client(handler).build(b"t", ["check"])
    assert report.compiled is False
    summary = compile_summary(report)
    assert summary["success"] is False
    assert summary["errors"] == ["src/lib.rs:2:21: error[E0308]: mismatched types"]
    assert summary["warnings"] == ["src/lib.rs:5: warning: unused variable: `x`"]
    assert summary["timing_ms"] == 800


async def test_compile_summary_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": {"build": {
            "exit": None, "timed_out": True, "ms": 120000, "stdout": "", "stderr": "",
        }}})

    summary = compile_summary(await _client(handler).build(b"t", ["build"]))
    assert summary["success"] is False
    assert summary["errors"] == ["compile timed out"]


async def test_non_200_raises_build_service_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "no_cargo_toml"})

    with pytest.raises(BuildServiceError, match="400"):
        await _client(handler).build(b"t", ["check"])


async def test_unreachable_service_raises_build_service_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(BuildServiceError, match="unreachable"):
        await _client(handler).build(b"t", ["check"])


async def test_compile_summary_no_compile_command() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": {"clippy": {
            "exit": 0, "timed_out": False, "ms": 10, "stdout": "", "stderr": "",
        }}})

    summary = compile_summary(await _client(handler).build(b"t", ["clippy"]))
    assert summary["success"] is False
    assert "no compile command" in summary["errors"][0]
