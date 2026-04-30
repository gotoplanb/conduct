"""Prompt resolution: walk client override → shared, capture git lineage.

Reads from disk on every resolve so prompt edits hot-reload without restart.
The git-hash subprocess is cached briefly to avoid forking per request.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_HASH_TTL_S = 5.0


class PromptNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class ResolvedPrompt:
    content: str
    path: str  # repo-relative, e.g. "prompts/shared/bio_generation.md"
    git_hash: str | None  # commit SHA of last commit touching this file, or None


class PromptResolver:
    def __init__(
        self,
        base_dir: Path = DEFAULT_PROMPTS_DIR,
        hash_ttl_s: float = DEFAULT_HASH_TTL_S,
    ) -> None:
        self.base_dir = base_dir
        self.repo_root = base_dir.parent
        self.hash_ttl_s = hash_ttl_s
        self._hash_cache: dict[str, tuple[float, str | None]] = {}

    def resolve(self, task_type: str, client_name: str | None = None) -> ResolvedPrompt:
        candidates: list[Path] = []
        if client_name:
            candidates.append(self.base_dir / "clients" / client_name / f"{task_type}.md")
        candidates.append(self.base_dir / "shared" / f"{task_type}.md")
        for path in candidates:
            if path.is_file():
                rel = str(path.relative_to(self.repo_root))
                return ResolvedPrompt(
                    content=path.read_text(encoding="utf-8"),
                    path=rel,
                    git_hash=self._git_hash(rel),
                )
        raise PromptNotFoundError(
            f"no prompt found for task_type={task_type!r}, client={client_name!r}"
        )

    def _git_hash(self, rel_path: str) -> str | None:
        now = time.monotonic()
        cached = self._hash_cache.get(rel_path)
        if cached and now - cached[0] < self.hash_ttl_s:
            return cached[1]
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H", "--", rel_path],
                capture_output=True,
                text=True,
                cwd=self.repo_root,
                check=False,
                timeout=2.0,
            )
            sha = result.stdout.strip() if result.returncode == 0 else ""
            value = sha or None
        except (subprocess.SubprocessError, OSError):
            value = None
        self._hash_cache[rel_path] = (now, value)
        return value


_resolver: PromptResolver | None = None


def get_prompt_resolver() -> PromptResolver:
    global _resolver
    if _resolver is None:
        _resolver = PromptResolver()
    return _resolver
