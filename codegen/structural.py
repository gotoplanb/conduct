"""Structural / AST regression checks (#29).

A small AST-level scan of generated Rust for the failure modes that compile and
pass tests but are still wrong-shaped: unexpected ``unsafe``, happy-path panics
(``unwrap``/``expect``/``panic!``/``todo!``/…), and drift from a required
function signature. Pairs with ``clippy`` (run in the build sandbox) in the
``code_eval`` ``structural`` dimension.

The parser is deliberately behind a single ``analyze_source`` boundary
([Agnostic]) so a ``syn``-based or different grammar can be swapped in without
touching the evaluator. v1 uses tree-sitter-rust.
"""

from __future__ import annotations

import io
import tarfile

import tree_sitter_rust
from tree_sitter import Language, Parser

_RUST = Language(tree_sitter_rust.language())
# Method calls that panic on the unhappy path (the signature implies totality).
_PANIC_METHODS = frozenset({"unwrap", "expect"})
# Macros that abort / mark incompleteness.
_PANIC_MACROS = frozenset({"panic", "unreachable", "todo", "unimplemented"})


def _walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


def _text(node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", "replace")


def _fn_signature(node, src: bytes) -> str:
    """A function_item's signature: text up to the body block, whitespace-collapsed."""
    body = next((c for c in node.children if c.type == "block"), None)
    end = body.start_byte if body else node.end_byte
    return src[node.start_byte : end].decode("utf-8", "replace").strip()


def _macro_name(node, src: bytes) -> str:
    ident = next((c for c in node.children if c.type in ("identifier", "scoped_identifier")), None)
    return _text(ident, src) if ident else ""


def _call_method(node, src: bytes) -> str:
    """For a call_expression, the method name if it's a `.method(...)` call."""
    func = node.child_by_field_name("function")
    if func is not None and func.type == "field_expression":
        field = func.child_by_field_name("field")
        if field is not None:
            return _text(field, src)
    return ""


def _signature_drift(found: list[str], required: list[str]) -> list[str]:
    """Required signatures (whitespace-insensitive) not present in any found fn.
    Lenient on additions — a found sig that *extends* the required one matches."""
    norm_found = ["".join(s.split()) for s in found]
    missing = []
    for req in required:
        nreq = "".join(req.split())
        if not any(nreq in f for f in norm_found):
            missing.append(req)
    return missing


def analyze_source(
    source: str, required_signatures: list[str] | None = None, *, file: str = ""
) -> dict:
    """AST scan of one Rust source. Returns unsafe / happy-path-panic sites,
    missing required signatures, a total violation count, and `ok`."""
    src = source.encode("utf-8")
    tree = Parser(_RUST).parse(src)
    unsafe, panics, sigs = [], [], []
    for n in _walk(tree.root_node):
        line = n.start_point[0] + 1
        if n.type == "unsafe_block":
            unsafe.append({"file": file, "line": line, "kind": "unsafe_block"})
        elif n.type == "function_modifiers" and "unsafe" in _text(n, src):
            unsafe.append({"file": file, "line": line, "kind": "unsafe_fn"})
        elif n.type == "function_item":
            sigs.append(_fn_signature(n, src))
        elif n.type == "call_expression" and (m := _call_method(n, src)) in _PANIC_METHODS:
            panics.append({"file": file, "line": line, "kind": m})
        elif n.type == "macro_invocation" and (mac := _macro_name(n, src)) in _PANIC_MACROS:
            panics.append({"file": file, "line": line, "kind": f"{mac}!"})
    drift = _signature_drift(sigs, required_signatures or [])
    violations = len(unsafe) + len(panics) + len(drift)
    return {
        "unsafe": unsafe, "panics": panics, "signature_drift": drift,
        "violations": violations, "ok": violations == 0,
    }


def read_rust_sources_from_tar(tar_bytes: bytes) -> dict[str, str]:
    """Every .rs file in a project tarball, keyed by path."""
    out: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith(".rs"):
                f = tar.extractfile(m)
                if f:
                    out[m.name] = f.read().decode("utf-8", "replace")
    return out


def analyze_tar(tar_bytes: bytes, required_signatures: list[str] | None = None) -> dict:
    """Aggregate analyze_source over every .rs file in the crate. Signature drift
    is computed against the union of all functions (a required sig satisfied in
    any file is not drift)."""
    sources = read_rust_sources_from_tar(tar_bytes)
    unsafe, panics, all_sigs = [], [], []
    for path, src in sorted(sources.items()):
        one = analyze_source(src, file=path)
        unsafe += one["unsafe"]
        panics += one["panics"]
        all_sigs.append(src)
    drift = _signature_drift(
        # re-extract signatures across all files for a union check
        [s for src in all_sigs for s in _all_fn_signatures(src)],
        required_signatures or [],
    )
    violations = len(unsafe) + len(panics) + len(drift)
    return {
        "files": sorted(sources), "unsafe": unsafe, "panics": panics,
        "signature_drift": drift, "violations": violations, "ok": violations == 0,
    }


def _all_fn_signatures(source: str) -> list[str]:
    src = source.encode("utf-8")
    tree = Parser(_RUST).parse(src)
    return [_fn_signature(n, src) for n in _walk(tree.root_node) if n.type == "function_item"]
