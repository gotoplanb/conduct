"""Client for the Watchtower rust-build sandbox (#24, gotoplanb/watchtower#1).

Transport + contract only: ship a Cargo artifact tarball (#23) plus a list of
logical commands to the local build service and return its raw structured
results. Interpretation into scoring dimensions lives in the ``code_eval``
evaluator (#25), not here.

Local-only: the build service runs in Watchtower's compose; there is no cloud
equivalent (code generation is a local research capability — see the README's
"Two purposes"). On a cloud worker, ``rust_build_url`` simply won't resolve and
a code_eval would fail cleanly with :class:`BuildServiceError`.

This module owns the authoritative request/response contract the Watchtower
service implements:

    POST /build {"project_tar_b64", "commands": [...], "timeout_s"?}
    -> {"results": {cmd: {"exit", "timed_out", "ms", "stdout", "stderr"}}}
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# Logical commands the runner understands (it owns the cargo argv mapping).
COMMANDS = ("check", "build", "clippy", "test")


@dataclass
class CommandResult:
    command: str
    exit_code: int | None
    timed_out: bool
    ms: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass
class BuildReport:
    results: dict[str, CommandResult] = field(default_factory=dict)

    def get(self, command: str) -> CommandResult | None:
        return self.results.get(command)

    @property
    def compiled(self) -> bool:
        """True if the project compiled — prefers `build`, falls back to `check`."""
        r = self.results.get("build") or self.results.get("check")
        return bool(r and r.ok)


class BuildServiceError(RuntimeError):
    """The build service was unreachable or returned a non-200 envelope. The
    calling job should fail cleanly rather than record a bogus result."""


class RustBuildClient:
    """Thin httpx client for the rust-build sandbox. Mirrors the media-provider
    construction pattern (base_url + timeout). ``transport`` is injectable so
    tests can stub the HTTP layer without a live service."""

    def __init__(
        self, base_url: str, *, timeout_s: float = 180.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._transport = transport

    async def build(
        self, tar_bytes: bytes, commands: list[str], *, timeout_s: int | None = None,
        overlay_files: dict | None = None, test_target: str | None = None,
    ) -> BuildReport:
        payload: dict = {
            "project_tar_b64": base64.b64encode(tar_bytes).decode(),
            "commands": commands,
        }
        if timeout_s is not None:
            payload["timeout_s"] = timeout_s
        if overlay_files:
            payload["overlay_files"] = overlay_files
        if test_target:
            payload["test_target"] = test_target
        try:
            async with httpx.AsyncClient(
                base_url=self._base, timeout=self._timeout_s, transport=self._transport
            ) as client:
                resp = await client.post("/build", json=payload)
        except httpx.HTTPError as e:
            raise BuildServiceError(f"build service unreachable: {e}") from e
        if resp.status_code != 200:
            raise BuildServiceError(f"build service error {resp.status_code}: {resp.text[:300]}")
        return _parse_report(resp.json())

    async def build_artifact(
        self, tar_path: str | Path, commands: list[str], *, timeout_s: int | None = None
    ) -> BuildReport:
        """Convenience: read a stored artifact tarball (#23) and build it."""
        return await self.build(Path(tar_path).read_bytes(), commands, timeout_s=timeout_s)


def _parse_report(body: dict) -> BuildReport:
    results: dict[str, CommandResult] = {}
    for cmd, r in (body.get("results") or {}).items():
        results[cmd] = CommandResult(
            command=cmd,
            exit_code=r.get("exit"),
            timed_out=bool(r.get("timed_out")),
            ms=int(r.get("ms") or 0),
            stdout=r.get("stdout") or "",
            stderr=r.get("stderr") or "",
        )
    return BuildReport(results=results)


def _split_cargo_diagnostics(stderr: str) -> tuple[list[str], list[str]]:
    """Split `cargo --message-format=short` stderr into error / warning lines.

    The short format prefixes the path, e.g. `src/lib.rs:2:21: error[E0308]: ...`
    and `src/lib.rs:5: warning: ...`, plus bare summary lines like
    `error: could not compile ...` — so match the diagnostic token anywhere,
    not just at line start."""
    errors, warnings = [], []
    for line in stderr.splitlines():
        s = line.strip()
        low = s.lower()
        if low.startswith("error") or "error[" in low or ": error" in low:
            errors.append(s)
        elif low.startswith("warning") or ": warning" in low or "warning:" in low:
            warnings.append(s)
    return errors, warnings


_TEST_RESULT_RE = re.compile(r"test result:\s+\w+\.\s+(\d+)\s+passed;\s+(\d+)\s+failed")
_COUNTEREX_RE = re.compile(r"minimal failing input:\s*(.+)")


def summarize_tests(report: BuildReport, command: str = "test") -> dict:
    """Parse `cargo test` output into a pass-rate (#26). Sums all `test result:`
    lines (a run can have several). `ok` requires a clean exit with >0 tests and
    no failures; a test binary that won't compile yields total=0 → not ok."""
    r = report.results.get(command)
    if r is None:
        return {"ran": False, "passed": 0, "failed": 0, "total": 0,
                "pass_rate": 0.0, "ok": False, "timed_out": False}
    if r.timed_out:
        return {"ran": True, "passed": 0, "failed": 0, "total": 0,
                "pass_rate": 0.0, "ok": False, "timed_out": True}
    text = f"{r.stdout}\n{r.stderr}"
    passed = failed = 0
    for m in _TEST_RESULT_RE.finditer(text):
        passed += int(m.group(1))
        failed += int(m.group(2))
    total = passed + failed
    return {
        "ran": True, "passed": passed, "failed": failed, "total": total,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "ok": r.exit_code == 0 and failed == 0 and total > 0,
        "timed_out": False,
    }


def counterexample(report: BuildReport, command: str = "test") -> str | None:
    """The proptest minimized failing input, if present in the test output."""
    r = report.results.get(command)
    if r is None:
        return None
    m = _COUNTEREX_RE.search(f"{r.stdout}\n{r.stderr}")
    return m.group(1).strip() if m else None


def compile_summary(report: BuildReport) -> dict:
    """The compile gate as a flat result (#24 acceptance):
    ``{success, errors[], warnings[], timing_ms}``. Prefers `build` over `check`.
    The code_eval evaluator (#25) turns this into a scoring dimension."""
    r = report.results.get("build") or report.results.get("check")
    if r is None:
        return {
            "success": False, "errors": ["no compile command run"],
            "warnings": [], "timing_ms": 0,
        }
    if r.timed_out:
        return {
            "success": False, "errors": ["compile timed out"],
            "warnings": [], "timing_ms": r.ms,
        }
    errors, warnings = _split_cargo_diagnostics(r.stderr)
    return {"success": r.ok, "errors": errors, "warnings": warnings, "timing_ms": r.ms}
