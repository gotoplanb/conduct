"""Unit tests for codegen.structural — the AST regression checks (#29)."""

from __future__ import annotations

import io
import tarfile

from codegen.structural import analyze_source, analyze_tar

_CLEAN = "pub fn add(a: i64, b: i64) -> i64 { a + b }\n"


def test_clean_source_has_no_violations() -> None:
    r = analyze_source(_CLEAN, required_signatures=["pub fn add(a: i64, b: i64) -> i64"])
    assert r["ok"] is True and r["violations"] == 0


def test_flags_unsafe_block_and_fn() -> None:
    src = "pub fn f() { unsafe { let _p = 0 as *const i32; } }\nunsafe fn g() {}\n"
    r = analyze_source(src)
    kinds = sorted(u["kind"] for u in r["unsafe"])
    assert kinds == ["unsafe_block", "unsafe_fn"]
    assert r["ok"] is False


def test_flags_happy_path_panics() -> None:
    src = (
        'pub fn f() -> i64 {\n'
        '    let v = std::env::var("X").unwrap();\n'
        '    let n: i64 = v.parse().expect("num");\n'
        '    if n < 0 { panic!("neg"); }\n'
        '    todo!()\n'
        '}\n'
    )
    kinds = sorted(p["kind"] for p in analyze_source(src)["panics"])
    assert kinds == ["expect", "panic!", "todo!", "unwrap"]


def test_signature_drift_flags_missing_required() -> None:
    src = "pub fn renamed(a: i64) -> i64 { a }\n"  # spec wanted `add`
    r = analyze_source(src, required_signatures=["pub fn add(a: i64, b: i64) -> i64"])
    assert r["signature_drift"] == ["pub fn add(a: i64, b: i64) -> i64"]


def test_signature_match_is_whitespace_insensitive() -> None:
    src = "pub fn add(a:i64,b:i64)->i64{a+b}\n"  # no spaces
    r = analyze_source(src, required_signatures=["pub fn add(a: i64, b: i64) -> i64"])
    assert r["signature_drift"] == []  # matches despite spacing


def test_signature_match_lenient_on_additions() -> None:
    # A found sig that extends the required one (e.g. a where-clause) still matches.
    src = "pub fn add(a: i64, b: i64) -> i64 where i64: Copy { a + b }\n"
    r = analyze_source(src, required_signatures=["pub fn add(a: i64, b: i64) -> i64"])
    assert r["signature_drift"] == []


def _tar(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_analyze_tar_aggregates_across_files() -> None:
    tar = _tar({
        "Cargo.toml": "[package]\nname='x'\n",
        "src/lib.rs": "pub fn add(a: i64, b: i64) -> i64 { a + b }\n",
        "src/util.rs": "pub fn helper() { unsafe { } }\n",
    })
    r = analyze_tar(tar, required_signatures=["pub fn add(a: i64, b: i64) -> i64"])
    assert r["files"] == ["src/lib.rs", "src/util.rs"]
    assert len(r["unsafe"]) == 1  # from util.rs
    assert r["signature_drift"] == []  # add() is present in lib.rs (union check)
    assert r["ok"] is False
