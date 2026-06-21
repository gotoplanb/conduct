"""Materialize a generated Rust Cargo project from a model's text output and
store it as a deterministic tarball artifact (#23).

The model's output is parsed into ``{path: content}`` via one of two accepted
contracts (a JSON ``{"files": {...}}`` manifest, or fenced code blocks tagged
with a path), validated (must contain ``Cargo.toml``; reject path traversal,
absolute paths, and oversize output), then written to ``{output_dir}/{id}.tar``.

Malformed output raises :class:`ArtifactError` with a stable ``reason`` code so
the caller can record a structured failure rather than crash — the same stance
the judge verdict parser takes.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

# A generated crate is small; these bound runaway / abusive output.
MAX_FILES = 64
MAX_FILE_BYTES = 256 * 1024  # 256 KiB per file
MAX_TOTAL_BYTES = 2 * 1024 * 1024  # 2 MiB whole project

# ```<info>\n<body>\n``` — info may carry a language and/or a path.
_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)\n?```", re.DOTALL)
# A `// file: <path>` or `# file: <path>` marker as a block's first line.
_FILE_MARKER_RE = re.compile(r"^\s*(?://|#)\s*file:\s*(.+?)\s*$")
_CARGO_TOML = "Cargo.toml"
# Bare root files we accept without an extension-bearing path.
_BARE_ROOT_FILES = (_CARGO_TOML, "Cargo.lock")


class ArtifactError(ValueError):
    """A structured, non-crashing failure to extract a code artifact.

    ``reason`` is a stable machine code (``missing_cargo_toml``,
    ``invalid_path``, ``no_files``, ``artifact_too_large``, ...); ``detail`` is
    free-text context. Subclasses ValueError so callers already catching
    ValueError on parse paths handle it uniformly.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


def _looks_like_path(tok: str) -> str | None:
    """Return a cleaned path if ``tok`` looks like a file path, else None.

    Accepts bare ``Cargo.toml``/``Cargo.lock`` and anything ending in a
    ``name.ext`` (optionally with directories). Rejects bare language tokens
    like ``rust`` (no extension)."""
    tok = tok.strip().strip("`*").strip()
    if not tok or " " in tok or "\n" in tok:
        return None
    if tok in _BARE_ROOT_FILES:
        return tok
    if re.fullmatch(r"[\w./\-]+\.\w+", tok):
        return tok
    return None


def _path_from_info(info: str) -> str | None:
    for tok in info.split():
        path = _looks_like_path(tok)
        if path:
            return path
    return None


def _path_from_body(body: str) -> str | None:
    first = body.split("\n", 1)[0]
    m = _FILE_MARKER_RE.match(first)
    return m.group(1).strip() if m else None


def _path_from_preceding(text: str, idx: int) -> str | None:
    """A path on the last non-empty line before the fence — e.g. ``**Cargo.toml**``,
    `` `src/main.rs` ``, or ``File: src/main.rs``."""
    prefix = text[:idx].rstrip()
    if not prefix:
        return None
    last = prefix.rsplit("\n", 1)[-1].strip()
    if last.lower().startswith("file:"):
        last = last[len("file:") :].strip()
    return _looks_like_path(last)


def _strip_file_marker(body: str) -> str:
    """Drop a leading ``// file:`` / ``# file:`` marker line from a block body."""
    first, sep, rest = body.partition("\n")
    return rest if (sep and _FILE_MARKER_RE.match(first)) else body


def _strip_leading_path_marker(body: str, resolved_path: str) -> str:
    """Drop a leading bare-path line from a block body when it equals the
    resolved path. This is the gemma4:12b shape surfaced by conduct#38:

        ```rust
        src/lib.rs
        pub fn ...   <- actual source
        ```

    The fence is untagged, so the path is inferred from language convention
    (or a preceding bold/backtick path). The model then ALSO echoes the path
    as the body's first line, which — unlike the ``// file:`` / ``# file:``
    marker — has no comment prefix, so the existing :func:`_strip_file_marker`
    leaves it in place and rustc fails on ``src/lib.rs:1:4: expected one of
    `!` or `::`, found `/` ``.

    The strip is gated on the first line matching ``resolved_path`` exactly
    so we never mangle source code whose first line merely looks like a
    path. Comparison is whitespace-trimmed on both sides to absorb the
    ``**path**`` / `` `path` `` bolding that sometimes survives into the
    body when the model wraps the marker line in markdown emphasis.
    """
    if not resolved_path:
        return body
    first, sep, rest = body.partition("\n")
    if not sep:
        return body
    return rest if first.strip() == resolved_path else body


def _loads_tolerant(blob: str, max_fixes: int = 16):
    """json.loads, but tolerant of trailing commas — a very common LLM-JSON
    quirk (``{"a": 1,}``). On a decode error we look at the exact offending
    position; if the char before it is a comma (a trailing comma the parser
    rejected), we drop just that comma and retry. Only the parser-flagged
    position is touched, so commas inside string values are never corrupted.
    Returns the parsed object, or None on any non-recoverable error."""
    for _ in range(max_fixes + 1):
        try:
            return json.loads(blob)
        except json.JSONDecodeError as e:
            # Python may point AT the comma (3.13+: "Illegal trailing comma")
            # or at the following } / ] (older). Check at e.pos, then walk back
            # over whitespace to find the offending comma.
            i = e.pos
            if i < len(blob) and blob[i] == ",":
                blob = blob[:i] + blob[i + 1 :]
                continue
            j = i - 1
            while j >= 0 and blob[j] in " \t\r\n":
                j -= 1
            if j >= 0 and blob[j] == ",":
                blob = blob[:j] + blob[j + 1 :]
                continue
            return None
    return None


def _try_json_manifest(text: str) -> dict[str, str] | None:
    """If the output's outermost ``{...}`` is a ``{"files": {path: content}}``
    manifest, return it. Otherwise None (fall back to fenced extraction).
    Tolerant of trailing commas (small models emit them often)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    obj = _loads_tolerant(text[start : end + 1])
    files = obj.get("files") if isinstance(obj, dict) else None
    if not isinstance(files, dict) or not files:
        return None
    out: dict[str, str] = {}
    for key, content in files.items():
        if not isinstance(content, str):
            raise ArtifactError("invalid_manifest", f"file {key!r} content is not a string")
        out[str(key)] = content
    return out


def _conventional_path(info: str, body: str, assigned: set[str]) -> str | None:
    """Infer a Cargo path from a fence's language alone, for the common case
    where a small model emits ```toml / ```rust blocks with no path tag. A toml
    block → Cargo.toml; the first rust block → src/main.rs if it has `fn main`,
    else src/lib.rs. Each conventional path is assigned at most once so a second
    untagged block of the same kind isn't clobbered (it just stays skipped)."""
    lang = info.split()[0].lower() if info.split() else ""
    if lang == "toml" and _CARGO_TOML not in assigned:
        return _CARGO_TOML
    if lang in ("rust", "rs"):
        if "fn main" in body and "src/main.rs" not in assigned:
            return "src/main.rs"
        if "src/lib.rs" not in assigned:
            return "src/lib.rs"
    return None


def _extract_fenced_files(text: str) -> dict[str, str]:
    """Pull path-tagged fenced code blocks. An explicit path (fence info,
    `// file:` marker, or a preceding path line) wins; failing that, a single
    toml/rust block falls back to the conventional Cargo path. Blocks with no
    resolvable path (prose, examples) are skipped."""
    files: dict[str, str] = {}
    for m in _FENCE_RE.finditer(text):
        info, body = m.group(1).strip(), m.group(2)
        path = (
            _path_from_info(info)
            or _path_from_body(body)
            or _path_from_preceding(text, m.start())
            or _conventional_path(info, body, set(files))
        )
        if path is None:
            continue
        # Strip order: first any ``// file:`` / ``# file:`` marker (comment-style),
        # then a leading bare path that matches the resolved path (the gemma4:12b
        # untagged-fence shape from conduct#38). The second strip is gated on
        # equality with the path we just inferred so it can't mangle a source
        # whose first line merely happens to look like a path.
        content = _strip_file_marker(body)
        content = _strip_leading_path_marker(content, path)
        files[path] = content
    return files


def _validate_path(raw: str) -> str:
    """Normalize a relative POSIX path; reject absolute, ``..``, or empty
    components (path-traversal / zip-slip defense)."""
    p = raw.strip().replace("\\", "/")
    if not p:
        raise ArtifactError("invalid_path", "empty path")
    pp = PurePosixPath(p)
    if pp.is_absolute() or any(part in ("..", "") for part in pp.parts) or pp.parts == (".",):
        raise ArtifactError("invalid_path", p)
    return pp.as_posix()


def parse_cargo_project(text: str) -> dict[str, str]:
    """Parse a model's output into a validated ``{path: content}`` Cargo project.

    Raises :class:`ArtifactError` (never an uncaught exception) on any malformed
    input: empty output, no tagged files, a bad path, oversize, or a missing /
    empty ``Cargo.toml``."""
    if not text or not text.strip():
        raise ArtifactError("empty_output")
    raw = _try_json_manifest(text)
    if raw is None:
        raw = _extract_fenced_files(text)
    if not raw:
        raise ArtifactError("no_files", "no path-tagged code blocks or files manifest found")

    files: dict[str, str] = {}
    total = 0
    for path, content in raw.items():
        norm = _validate_path(path)
        size = len(content.encode("utf-8"))
        if size > MAX_FILE_BYTES:
            raise ArtifactError(
                "artifact_too_large", f"{norm} is {size} bytes (> {MAX_FILE_BYTES})"
            )
        files[norm] = content
        total += size
    if len(files) > MAX_FILES:
        raise ArtifactError("artifact_too_large", f"{len(files)} files (> {MAX_FILES})")
    if total > MAX_TOTAL_BYTES:
        raise ArtifactError("artifact_too_large", f"{total} bytes (> {MAX_TOTAL_BYTES})")
    if _CARGO_TOML not in files:
        raise ArtifactError("missing_cargo_toml")
    if not files[_CARGO_TOML].strip():
        raise ArtifactError("empty_cargo_toml")
    return files


def write_cargo_tar(files: dict[str, str], dest: str | Path) -> int:
    """Write ``files`` to a deterministic tarball at ``dest`` (sorted entries,
    zeroed mtime/owner) so the same project yields a byte-identical artifact.
    Returns total uncompressed content bytes."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with tarfile.open(dest, "w") as tar:
        for path in sorted(files):
            data = files[path].encode("utf-8")
            info = tarfile.TarInfo(name=path)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            tar.addfile(info, io.BytesIO(data))
            total += len(data)
    return total


def build_cargo_artifact(text: str, *, job_id, output_dir: str | Path) -> dict:
    """Parse + store a Cargo project from ``text``. Returns the artifact
    metadata (``format``, ``url``, ``files``, ``file_count``, ``bytes``) for
    Job.metadata. Raises :class:`ArtifactError` on malformed output."""
    files = parse_cargo_project(text)
    filename = f"{job_id}.tar"
    total = write_cargo_tar(files, Path(output_dir) / filename)
    return {
        "format": "cargo",
        "url": f"/output/{filename}",
        "files": sorted(files),
        "file_count": len(files),
        "bytes": total,
    }
