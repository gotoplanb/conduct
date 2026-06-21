"""Unit tests for codegen.artifact — parse a model's output into a Cargo
project, validate it, and write a deterministic tarball (#23). No DB / no FS
except the tmp_path tar tests."""

from __future__ import annotations

import tarfile

import pytest

from codegen.artifact import (
    ArtifactError,
    build_cargo_artifact,
    parse_cargo_project,
    write_cargo_tar,
)

_CARGO = '[package]\nname = "sol"\nversion = "0.1.0"\nedition = "2021"\n'
_MAIN = "fn main() { println!(\"hi\"); }\n"


def test_json_manifest() -> None:
    text = (
        'Here is the project:\n```json\n'
        + '{"files": {"Cargo.toml": ' + _q(_CARGO) + ', "src/main.rs": ' + _q(_MAIN) + '}}'
        + '\n```\n'
    )
    files = parse_cargo_project(text)
    assert set(files) == {"Cargo.toml", "src/main.rs"}
    assert files["src/main.rs"] == _MAIN


def test_json_manifest_tolerates_trailing_comma() -> None:
    # The qwen3.5:9b #36 case: valid manifest but a trailing comma after the
    # last file. strict json.loads rejects it; we recover.
    text = (
        '{"files": {"Cargo.toml": ' + _q(_CARGO) + ', "src/main.rs": ' + _q(_MAIN) + ',}}'
    )
    files = parse_cargo_project(text)
    assert set(files) == {"Cargo.toml", "src/main.rs"}


def test_json_manifest_tolerates_multiple_trailing_commas() -> None:
    text = (
        '{"files": {"Cargo.toml": ' + _q(_CARGO) + ', "src/main.rs": ' + _q(_MAIN) + ',},}'
    )
    assert set(parse_cargo_project(text)) == {"Cargo.toml", "src/main.rs"}


def test_trailing_comma_fix_preserves_commas_inside_strings() -> None:
    # A literal ",}" inside a code string must survive — only the structural
    # trailing comma is removed, never string content.
    lib = 'fn f() -> &\'static str { "a,}b" }\n'
    text = '{"files": {"Cargo.toml": ' + _q(_CARGO) + ', "src/lib.rs": ' + _q(lib) + ',}}'
    files = parse_cargo_project(text)
    assert files["src/lib.rs"] == lib  # the ",}" inside the string is intact


def test_fenced_info_string_path() -> None:
    text = f"```toml Cargo.toml\n{_CARGO}```\n\n```rust src/main.rs\n{_MAIN}```\n"
    files = parse_cargo_project(text)
    assert set(files) == {"Cargo.toml", "src/main.rs"}


def test_fenced_file_marker_first_line() -> None:
    text = (
        f"```toml\n// file: Cargo.toml\n{_CARGO}```\n"
        f"```rust\n// file: src/main.rs\n{_MAIN}```\n"
    )
    files = parse_cargo_project(text)
    assert set(files) == {"Cargo.toml", "src/main.rs"}
    # the marker line is stripped from the stored content
    assert "file:" not in files["src/main.rs"]


def test_fenced_preceding_bold_path() -> None:
    text = f"**Cargo.toml**\n```toml\n{_CARGO}```\n\n`src/main.rs`\n```rust\n{_MAIN}```\n"
    files = parse_cargo_project(text)
    assert set(files) == {"Cargo.toml", "src/main.rs"}


def test_untagged_toml_rust_use_conventional_paths() -> None:
    # The common small-model shape: ```toml + ```rust with NO path tags.
    text = f"```toml\n{_CARGO}```\n\n```rust\n{_MAIN}```\n"
    files = parse_cargo_project(text)
    assert set(files) == {"Cargo.toml", "src/main.rs"}  # _MAIN has fn main


def test_untagged_rust_without_main_is_lib() -> None:
    lib = "pub fn add(a: i64, b: i64) -> i64 { a + b }\n"
    text = f"```toml\n{_CARGO}```\n```rust\n{lib}```\n"
    files = parse_cargo_project(text)
    assert set(files) == {"Cargo.toml", "src/lib.rs"}  # no fn main -> lib.rs


def test_explicit_path_beats_convention() -> None:
    # An explicit path tag wins over the language convention.
    lib = "pub fn f() {}\n"
    text = f"```toml Cargo.toml\n{_CARGO}```\n```rust src/thing.rs\n{lib}```\n"
    files = parse_cargo_project(text)
    assert set(files) == {"Cargo.toml", "src/thing.rs"}


def test_untagged_rust_with_bare_path_first_line_lib() -> None:
    # conduct#38: gemma4:12b emits an untagged ```rust fence whose first line
    # is the bare path "src/lib.rs". Without stripping, that line ends up as
    # line 1 of the source and rustc fails with "expected one of `!` or `::`,
    # found `/`". The conventional-path inference already resolves the path;
    # the body's leading path marker must be stripped.
    # The fence regex consumes the trailing \n before the closing ```, so the
    # body has no trailing newline once it's been captured.
    lib_body = "src/lib.rs\npub fn whatever() -> usize { 1 }"
    text = f"```toml\n{_CARGO}```\n```rust\n{lib_body}```\n"
    files = parse_cargo_project(text)
    assert set(files) == {"Cargo.toml", "src/lib.rs"}
    assert files["src/lib.rs"] == "pub fn whatever() -> usize { 1 }"
    # The path marker must not leak into the stored source.
    assert files["src/lib.rs"].splitlines()[0] == "pub fn whatever() -> usize { 1 }"


def test_untagged_rust_with_bare_path_first_line_main() -> None:
    # Same shape as the lib variant but with fn main — the conventional-path
    # fallback picks src/main.rs (via the fn-main detector) and the leading
    # bare path line still needs to be stripped.
    main_body = "src/main.rs\nfn main() { println!(\"hi\"); }"
    text = f"```toml\n{_CARGO}```\n```rust\n{main_body}```\n"
    files = parse_cargo_project(text)
    assert set(files) == {"Cargo.toml", "src/main.rs"}
    assert files["src/main.rs"] == "fn main() { println!(\"hi\"); }"
    assert files["src/main.rs"].splitlines()[0] == "fn main() { println!(\"hi\"); }"


def test_untagged_fence_path_marker_only_stripped_when_matches_resolved() -> None:
    # Defensive: a leading line that LOOKS like a path but doesn't match the
    # resolved path must NOT be stripped (it might be real code).
    # Here the fence resolves to src/lib.rs (no fn main); the body's first line
    # is "src/main.rs" which is a different path and must be preserved.
    body = "src/main.rs\npub fn kept() {}"
    text = f"```toml\n{_CARGO}```\n```rust\n{body}```\n"
    files = parse_cargo_project(text)
    assert set(files) == {"Cargo.toml", "src/lib.rs"}
    # The body was stored verbatim — the line that happened to look like a
    # path but didn't match the resolved path stayed.
    assert files["src/lib.rs"] == body


def test_lone_rust_snippet_fails_on_missing_cargo() -> None:
    # A bare ```rust snippet becomes src/lib.rs by convention, then fails for
    # lack of a Cargo.toml — still a structured failure, not a crash.
    text = "Here's a sketch:\n```rust\nfn foo() {}\n```\nNo project though.\n"
    with pytest.raises(ArtifactError) as e:
        parse_cargo_project(text)
    assert e.value.reason == "missing_cargo_toml"


def test_prose_with_no_code_is_no_files() -> None:
    # Genuinely no code blocks at all -> no_files.
    with pytest.raises(ArtifactError) as e:
        parse_cargo_project("Just some prose, no code at all.")
    assert e.value.reason == "no_files"


def test_missing_cargo_toml() -> None:
    text = f"```rust src/main.rs\n{_MAIN}```\n"
    with pytest.raises(ArtifactError) as e:
        parse_cargo_project(text)
    assert e.value.reason == "missing_cargo_toml"


def test_empty_cargo_toml() -> None:
    text = f"```toml Cargo.toml\n\n```\n```rust src/main.rs\n{_MAIN}```\n"
    with pytest.raises(ArtifactError) as e:
        parse_cargo_project(text)
    assert e.value.reason == "empty_cargo_toml"


def test_empty_output() -> None:
    with pytest.raises(ArtifactError) as e:
        parse_cargo_project("   \n  ")
    assert e.value.reason == "empty_output"


@pytest.mark.parametrize("bad", ["../evil.rs", "/etc/passwd.rs", "src/../../x.rs"])
def test_path_traversal_rejected_in_fence(bad: str) -> None:
    text = f"```toml Cargo.toml\n{_CARGO}```\n```rust {bad}\n{_MAIN}```\n"
    with pytest.raises(ArtifactError) as e:
        parse_cargo_project(text)
    assert e.value.reason == "invalid_path"


@pytest.mark.parametrize("bad", ["/etc/passwd", "../../escape.rs", "a/../../b.rs"])
def test_path_traversal_rejected_in_manifest(bad: str) -> None:
    # JSON-manifest paths bypass the fence path-sniff, so _validate_path is the
    # sole gate — including for extension-less absolute paths.
    import json

    text = json.dumps({"files": {"Cargo.toml": _CARGO, bad: _MAIN}})
    with pytest.raises(ArtifactError) as e:
        parse_cargo_project(text)
    assert e.value.reason == "invalid_path"


def test_oversize_file_rejected() -> None:
    big = "x" * (256 * 1024 + 1)
    text = f"```toml Cargo.toml\n{_CARGO}```\n```rust src/main.rs\n{big}\n```\n"
    with pytest.raises(ArtifactError) as e:
        parse_cargo_project(text)
    assert e.value.reason == "artifact_too_large"


def test_manifest_non_string_content_rejected() -> None:
    text = '{"files": {"Cargo.toml": 123}}'
    with pytest.raises(ArtifactError) as e:
        parse_cargo_project(text)
    assert e.value.reason == "invalid_manifest"


def test_write_tar_is_deterministic_and_roundtrips(tmp_path) -> None:
    files = {"Cargo.toml": _CARGO, "src/main.rs": _MAIN}
    a, b = tmp_path / "a.tar", tmp_path / "b.tar"
    write_cargo_tar(files, a)
    write_cargo_tar(files, b)
    assert a.read_bytes() == b.read_bytes()  # deterministic (sorted, mtime 0)

    with tarfile.open(a) as tar:
        names = sorted(tar.getnames())
        assert names == ["Cargo.toml", "src/main.rs"]
        extracted = tar.extractfile("src/main.rs").read().decode()
        assert extracted == _MAIN


def test_build_cargo_artifact_writes_and_returns_meta(tmp_path) -> None:
    text = f"```toml Cargo.toml\n{_CARGO}```\n```rust src/main.rs\n{_MAIN}```\n"
    jid = "11111111-1111-1111-1111-111111111111"
    meta = build_cargo_artifact(text, job_id=jid, output_dir=tmp_path)
    assert meta["format"] == "cargo"
    assert meta["url"] == "/output/11111111-1111-1111-1111-111111111111.tar"
    assert meta["files"] == ["Cargo.toml", "src/main.rs"]
    assert meta["file_count"] == 2 and meta["bytes"] > 0
    assert (tmp_path / "11111111-1111-1111-1111-111111111111.tar").is_file()


def _q(s: str) -> str:
    import json

    return json.dumps(s)
