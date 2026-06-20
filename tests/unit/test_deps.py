"""Unit tests for codegen.deps — the hallucinated-dependency detector (#27).

The crates.io sparse index is stubbed with httpx.MockTransport (no network);
the live registry is exercised in the code_eval validation.
"""

from __future__ import annotations

import io
import tarfile

import httpx
import pytest

from codegen.deps import (
    CratesIndexClient,
    Dep,
    _index_path,
    _req_satisfiable,
    check_dependencies,
    parse_cargo_dependencies,
    read_cargo_toml_from_tar,
    run_deps_check,
)

_CARGO = """
[package]
name = "x"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = "1.0"
rand = { version = "0.8", features = ["small_rng"] }
local_thing = { path = "../local_thing" }
gitdep = { git = "https://example.com/g.git" }

[dev-dependencies]
proptest = "1"
"""


def test_parse_cargo_dependencies_classifies_registry_vs_local() -> None:
    deps = {d.name: d for d in parse_cargo_dependencies(_CARGO)}
    assert deps["serde"].registry is True and deps["serde"].req == "1.0"
    assert deps["rand"].registry is True and deps["rand"].req == "0.8"
    assert deps["local_thing"].registry is False  # path dep, skipped
    assert deps["gitdep"].registry is False  # git dep, skipped
    assert deps["proptest"].kind == "dev-dependencies"


@pytest.mark.parametrize(
    ("name", "path"),
    [("a", "1/a"), ("ab", "2/ab"), ("abc", "3/a/abc"), ("serde", "se/rd/serde"),
     ("Tokio", "to/ki/tokio")],
)
def test_index_path(name: str, path: str) -> None:
    assert _index_path(name) == path


def test_req_satisfiable() -> None:
    assert _req_satisfiable("1.0", ["1.0.130", "1.2.0"]) is True
    assert _req_satisfiable("99", ["1.0.130"]) is False  # impossible major
    assert _req_satisfiable("0.8", ["0.8.5", "0.7.0"]) is True
    assert _req_satisfiable("0.9", ["0.8.5"]) is False  # 0.x minor absent
    # ambiguous reqs are assumed satisfiable (never false-flag)
    assert _req_satisfiable(">=1, <3", ["2.0.0"]) is True
    assert _req_satisfiable("1.*", ["1.4.0"]) is True


def _index(mapping: dict[str, list[str] | None]) -> CratesIndexClient:
    """mapping: crate-name -> versions list, or None to 404."""
    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        if name not in mapping or mapping[name] is None:
            return httpx.Response(404)
        body = "\n".join(
            f'{{"name":"{name}","vers":"{v}","yanked":false}}' for v in mapping[name]
        )
        return httpx.Response(200, text=body)

    return CratesIndexClient(transport=httpx.MockTransport(handler))


async def test_check_dependencies_flags_fabricated_crate() -> None:
    deps = [
        Dep("serde", "1.0", "dependencies", True),
        Dep("totally_made_up_xyz", "1.0", "dependencies", True),
        Dep("local", None, "dependencies", False),  # skipped
    ]
    client = _index({"serde": ["1.0.130"], "totally_made_up_xyz": None})
    result = await check_dependencies(deps, client)
    assert result["ok"] is False
    assert result["checked"] == 2  # the path dep is not checked
    assert result["offenders"] == [
        {"name": "totally_made_up_xyz", "reason": "not_found", "req": "1.0"}
    ]


async def test_check_dependencies_flags_impossible_version() -> None:
    deps = [Dep("serde", "99.0", "dependencies", True)]
    result = await check_dependencies(deps, _index({"serde": ["1.0.130", "1.2.0"]}))
    assert result["offenders"][0]["reason"] == "no_matching_version"


async def test_check_dependencies_all_real_is_clean() -> None:
    deps = [Dep("serde", "1.0", "dependencies", True), Dep("rand", "0.8", "dependencies", True)]
    result = await check_dependencies(deps, _index({"serde": ["1.0.130"], "rand": ["0.8.5"]}))
    assert result["ok"] is True and result["offenders"] == []


async def test_unreachable_crate_is_unresolved_not_offender() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = CratesIndexClient(transport=httpx.MockTransport(handler))
    result = await check_dependencies([Dep("serde", "1.0", "dependencies", True)], client)
    assert result["ok"] is True  # unknown != invalid
    assert result["unresolved"] == ["serde"]


def _tar(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


async def test_run_deps_check_reads_cargo_from_tar() -> None:
    tar = _tar({"Cargo.toml": _CARGO, "src/lib.rs": "pub fn f(){}"})
    assert read_cargo_toml_from_tar(tar).startswith("\n[package]")
    result = await run_deps_check(tar, _index({
        "serde": ["1.0.130"], "rand": ["0.8.5"], "proptest": ["1.4.0"],
    }))
    assert result["ok"] is True and result["checked"] == 3


async def test_run_deps_check_no_cargo_toml_returns_none() -> None:
    assert await run_deps_check(_tar({"src/lib.rs": "x"}), _index({})) is None


async def test_run_deps_check_unparseable_cargo_is_offender() -> None:
    result = await run_deps_check(_tar({"Cargo.toml": "this is not = = toml ["}), _index({}))
    assert result["ok"] is False
    assert result["offenders"][0]["reason"] == "unparseable"
