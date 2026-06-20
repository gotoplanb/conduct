"""Hallucinated-dependency detector (#27).

Parse a generated crate's ``Cargo.toml`` and resolve each registry dependency
against the crates.io **sparse index** (the CDN-backed source Cargo itself uses)
to flag fabricated/nonexistent crates and impossible version requirements —
*before* a build wastes a sandbox container.

This is a fast, network-only pre-build check that lives in the evaluator, not
the build sandbox: it is the authority on dependency validity, so the build can
stay offline-leaning without silently papering over a bad dep. Path/git/workspace
deps are skipped (they're not crates.io packages).
"""

from __future__ import annotations

import io
import json
import re
import tarfile
import tomllib
from dataclasses import dataclass
from pathlib import PurePosixPath

import httpx

_DEP_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")
_SEMVER_RE = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")
_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


class CratesIndexError(RuntimeError):
    """The crates.io index was unreachable / returned an unexpected status — the
    crate's validity is *unknown*, not invalid (so we never false-flag)."""


@dataclass
class Dep:
    name: str
    req: str | None  # version requirement, or None (e.g. a features-only table)
    kind: str
    registry: bool  # True only for crates.io deps (no path/git/workspace)


def _spec_to_req(spec) -> tuple[str | None, bool]:
    if isinstance(spec, str):
        return spec, True
    if isinstance(spec, dict):
        if spec.get("path") or spec.get("git") or spec.get("workspace"):
            return None, False
        return spec.get("version"), True
    return None, False


def parse_cargo_dependencies(toml_text: str) -> list[Dep]:
    """Pull registry dependencies from a Cargo.toml. Raises
    ``tomllib.TOMLDecodeError`` on a malformed file."""
    data = tomllib.loads(toml_text)
    deps: list[Dep] = []
    for kind in _DEP_TABLES:
        for name, spec in (data.get(kind) or {}).items():
            req, registry = _spec_to_req(spec)
            deps.append(Dep(name=str(name), req=req, kind=kind, registry=registry))
    return deps


def read_cargo_toml_from_tar(tar_bytes: bytes) -> str | None:
    """Extract the (first) Cargo.toml from a project tarball, or None."""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        member = next(
            (m for m in tar.getmembers() if PurePosixPath(m.name).name == "Cargo.toml"), None
        )
        if member is None:
            return None
        f = tar.extractfile(member)
        return f.read().decode("utf-8", "replace") if f else None


def _index_path(name: str) -> str:
    """crates.io sparse-index path for a crate name (lowercased)."""
    n = name.lower()
    if len(n) == 1:
        return f"1/{n}"
    if len(n) == 2:
        return f"2/{n}"
    if len(n) == 3:
        return f"3/{n[0]}/{n}"
    return f"{n[0:2]}/{n[2:4]}/{n}"


def _parse_version(v) -> tuple[int, int, int] | None:
    if not isinstance(v, str):
        return None
    core = v.split("+")[0].split("-")[0]
    m = _VERSION_RE.fullmatch(core)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def _leading_semver(req: str) -> tuple[int, int, int] | None:
    """Best-effort leading version of a simple req (bare/^/~/=). Returns None for
    ranges/wildcards/multi-reqs so we never false-flag an ambiguous requirement."""
    req = req.strip()
    if any(c in req for c in "*,<>|"):
        return None
    req = req.lstrip("^~=v ").strip()
    m = _SEMVER_RE.match(req)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0))


def _req_satisfiable(req: str, versions: list[str]) -> bool:
    """Conservative: only reports unsatisfiable when a clean caret/exact/tilde
    req's major (>=1) — or major.minor (0.x) — is absent from every published
    version. Ranges/wildcards are assumed satisfiable."""
    target = _leading_semver(req)
    if target is None:
        return True
    maj, minor, _ = target
    pubs = [p for p in (_parse_version(v) for v in versions) if p]
    if not pubs:
        return True
    if maj >= 1:
        return any(p[0] == maj for p in pubs)
    return any(p[0] == 0 and p[1] == minor for p in pubs)


class CratesIndexClient:
    """Resolves crate names against the crates.io sparse index. ``transport`` is
    injectable so tests stub the index without network."""

    def __init__(
        self, base_url: str = "https://index.crates.io", *, timeout_s: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._transport = transport

    async def versions(self, name: str) -> list[str] | None:
        """Published (non-yanked) versions of a crate, or None if it doesn't
        exist (404). Raises :class:`CratesIndexError` on transport/other errors."""
        try:
            async with httpx.AsyncClient(
                base_url=self._base, timeout=self._timeout_s, transport=self._transport
            ) as client:
                resp = await client.get(f"/{_index_path(name)}")
        except httpx.HTTPError as e:
            raise CratesIndexError(str(e)) from e
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise CratesIndexError(f"index returned {resp.status_code}")
        out = []
        for line in resp.text.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if not obj.get("yanked"):
                out.append(obj.get("vers"))
        return out


async def check_dependencies(deps: list[Dep], client: CratesIndexClient) -> dict:
    """Resolve each registry dep. Returns ``{checked, offenders[], unresolved[],
    ok}``. An offender names the specific crate + reason (``not_found`` /
    ``no_matching_version``). A crate we couldn't reach is ``unresolved`` (not an
    offender — its validity is unknown)."""
    offenders, unresolved, checked = [], [], 0
    for d in deps:
        if not d.registry:
            continue
        checked += 1
        try:
            versions = await client.versions(d.name)
        except CratesIndexError:
            unresolved.append(d.name)
            continue
        if versions is None:
            offenders.append({"name": d.name, "reason": "not_found", "req": d.req})
        elif d.req and not _req_satisfiable(d.req, versions):
            offenders.append({"name": d.name, "reason": "no_matching_version", "req": d.req})
    return {
        "checked": checked, "offenders": offenders,
        "unresolved": unresolved, "ok": not offenders,
    }


async def run_deps_check(tar_bytes: bytes, client: CratesIndexClient) -> dict | None:
    """Read the project's Cargo.toml from the tarball and resolve its deps.
    Returns the check result, or None if there's no Cargo.toml. Never raises —
    a malformed Cargo.toml is itself a (structured) offender."""
    toml_text = read_cargo_toml_from_tar(tar_bytes)
    if toml_text is None:
        return None
    try:
        deps = parse_cargo_dependencies(toml_text)
    except tomllib.TOMLDecodeError as e:
        return {
            "checked": 0, "unresolved": [], "ok": False,
            "offenders": [{"name": "Cargo.toml", "reason": "unparseable", "req": str(e)[:120]}],
        }
    return await check_dependencies(deps, client)
