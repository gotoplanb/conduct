"""Coverage gaps in codegen/artifact.py: tolerant-JSON edge cases, path
inference fallbacks, and the size/count ceilings."""

from __future__ import annotations

import json

import pytest

from codegen.artifact import (
    MAX_FILES,
    ArtifactError,
    _conventional_path,
    _loads_tolerant,
    parse_cargo_project,
)

_CARGO = '[package]\nname = "demo"\n'


def test_preceding_file_label_names_the_block() -> None:
    # A "File: <path>" line right before the fence is a valid path source.
    text = (
        f"```toml\n{_CARGO}```\n"
        "File: src/main.rs\n"
        "```rust\nfn main() {}\n```\n"
    )
    files = parse_cargo_project(text)
    assert set(files) == {"Cargo.toml", "src/main.rs"}


def test_loads_tolerant_truncated_json_gives_up_cleanly() -> None:
    # Truncated output: the error position is past the end, the walk-back finds
    # the dangling comma, and the retry still fails -> None, not a crash.
    assert _loads_tolerant('{"a": 1,') is None


def test_loads_tolerant_fix_budget_exhausted() -> None:
    # More broken commas than max_fixes -> give up after the budget.
    assert _loads_tolerant("[1" + "," * 20 + "]") is None


def test_conventional_path_exhausted_returns_none() -> None:
    # Both conventional rust slots assigned -> a third untagged block is skipped.
    assigned = {"src/main.rs", "src/lib.rs"}
    assert _conventional_path("rust", "fn main() {}", assigned) is None
    assert _conventional_path("text", "prose", set()) is None


def test_untagged_extra_blocks_are_skipped() -> None:
    # Third rust block has no explicit path and both conventional slots are
    # taken; a bare prose fence has no path either. Neither may land in files.
    text = (
        f"```toml\n{_CARGO}```\n"
        "```rust\nfn main() {}\n```\n"
        "```rust\npub fn lib() {}\n```\n"
        "```rust\npub fn extra() {}\n```\n"
        "```\njust prose, no path\n```\n"
    )
    files = parse_cargo_project(text)
    assert set(files) == {"Cargo.toml", "src/main.rs", "src/lib.rs"}


def test_blank_path_rejected() -> None:
    blob = json.dumps({"files": {" ": "x", "Cargo.toml": _CARGO}})
    with pytest.raises(ArtifactError, match="invalid_path"):
        parse_cargo_project(blob)


def test_too_many_files_rejected() -> None:
    manifest = {f"src/f{i}.rs": "// x" for i in range(MAX_FILES)}
    manifest["Cargo.toml"] = _CARGO  # MAX_FILES + 1 total
    with pytest.raises(ArtifactError, match="artifact_too_large") as exc:
        parse_cargo_project(json.dumps({"files": manifest}))
    assert "files" in exc.value.detail


def test_total_bytes_ceiling_rejected() -> None:
    # Every file under the per-file cap, but the sum crosses the 2 MiB total.
    big = "x" * 250_000
    manifest = {f"src/f{i}.rs": big for i in range(9)}
    manifest["Cargo.toml"] = _CARGO
    with pytest.raises(ArtifactError, match="artifact_too_large") as exc:
        parse_cargo_project(json.dumps({"files": manifest}))
    assert "bytes" in exc.value.detail
