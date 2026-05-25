"""`conduct` CLI — admin commands that talk to a running Conduct server.

Usage:
  conduct prompts list
  conduct prompts get <task_type> [--client <name>]
  conduct prompts edit <task_type> [--client <name>]
  conduct prompts history <task_type> [--client <name>] [--limit N]

Auth: reads `CONDUCT_ADMIN_KEY` (required for any non-trivial call).
Base URL: reads `CONDUCT_BASE_URL` (default http://localhost:8000).

`edit` is the headline feature — it round-trips through $EDITOR like
`git commit -e`: fetch current content, write it to a tempfile, open the
editor, then PUT the result back on save. If the file is unchanged or
empty, the edit is aborted with no DB write.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"


def _base_url() -> str:
    return os.environ.get("CONDUCT_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _admin_key() -> str:
    key = os.environ.get("CONDUCT_ADMIN_KEY", "").strip()
    if not key:
        print(
            "error: CONDUCT_ADMIN_KEY is not set. Export it with the admin key "
            "configured for your Conduct instance.",
            file=sys.stderr,
        )
        sys.exit(2)
    return key


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_admin_key()}"}


def _client_params(client: str | None) -> dict[str, str]:
    return {"client": client} if client else {}


def _editor() -> list[str]:
    """`$EDITOR` split for subprocess. Falls back to `vi` (POSIX default)."""
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    # Allow `EDITOR="code --wait"` and friends.
    return editor.split()


def _open_in_editor(initial: str, *, suffix: str = ".md") -> str | None:
    """Write `initial` to a tempfile, open $EDITOR on it, return the saved
    contents. Returns None if the file is empty or unchanged."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    ) as tf:
        tf.write(initial)
        tmp_path = Path(tf.name)
    try:
        result = subprocess.run([*_editor(), str(tmp_path)], check=False)
        if result.returncode != 0:
            print(f"editor exited with status {result.returncode} — aborting", file=sys.stderr)
            return None
        new_content = tmp_path.read_text(encoding="utf-8")
        if not new_content.strip():
            print("aborting: empty content", file=sys.stderr)
            return None
        if new_content == initial:
            print("no changes — nothing to save", file=sys.stderr)
            return None
        return new_content
    finally:
        tmp_path.unlink(missing_ok=True)


# --- subcommands ----


def cmd_prompts_list(args: argparse.Namespace) -> None:
    """Print all prompts. Errors raise via HTTPStatusError → handled in main()."""
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30) as c:
        r = c.get("/prompts")
        r.raise_for_status()
        data = r.json()
    rows = data.get("prompts", [])
    if not rows:
        print("(no prompts)")
        return
    width = max(len(p["task_type"]) for p in rows)
    for p in rows:
        client = p["client_name"] or "<shared>"
        print(f"{p['task_type']:<{width}}  {client:<24}  {p['bytes']:>6}B  {p['updated_at']}")


def cmd_prompts_get(args: argparse.Namespace) -> int:
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30) as c:
        r = c.get(f"/prompts/{args.task_type}", params=_client_params(args.client))
        if r.status_code == 404:
            print(r.json().get("detail", "not found"), file=sys.stderr)
            return 1
        r.raise_for_status()
        data = r.json()
    sys.stdout.write(data["content"])
    if not data["content"].endswith("\n"):
        sys.stdout.write("\n")
    return 0


def cmd_prompts_edit(args: argparse.Namespace) -> None:
    """Open $EDITOR on the current content, PUT the result on save. A 404
    from the GET means we're creating a new prompt — start with an empty
    buffer. Empty / unchanged buffers abort with no DB write."""
    base = _base_url()
    headers = _headers()
    params = _client_params(args.client)

    with httpx.Client(base_url=base, headers=headers, timeout=30) as c:
        r = c.get(f"/prompts/{args.task_type}", params=params)
        if r.status_code == 404:
            initial = ""
            print(
                f"note: no existing prompt for task_type={args.task_type!r} "
                f"client={args.client!r} — editing a new one",
                file=sys.stderr,
            )
        else:
            r.raise_for_status()
            initial = r.json()["content"]

        new_content = _open_in_editor(initial)
        if new_content is None:
            return

        put = c.put(
            f"/prompts/{args.task_type}",
            params=params,
            json={"content": new_content},
        )
        put.raise_for_status()
        out = put.json()
    target = f"client={args.client}" if args.client else "shared"
    print(
        f"saved prompts/{args.task_type} ({target}) — "
        f"{len(out['content'].encode('utf-8'))} bytes, updated_by={out['updated_by']}"
    )


def cmd_prompts_history(args: argparse.Namespace) -> int:
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30) as c:
        r = c.get(
            f"/prompts/{args.task_type}/history",
            params={**_client_params(args.client), "limit": str(args.limit)},
        )
        if r.status_code == 404:
            print(r.json().get("detail", "not found"), file=sys.stderr)
            return 1
        r.raise_for_status()
        data = r.json()
    versions = data.get("versions", [])
    if not versions:
        print("(no history)")
        return 0
    for v in versions:
        editor = v["edited_by"] or "<unknown>"
        print(f"{v['id']:>6}  {v['edited_at']}  by {editor:<14}  {v['bytes']:>6}B")
    return 0


def cmd_jobs_list(args: argparse.Namespace) -> None:
    """List recent jobs across all clients (admin), newest first."""
    params: dict[str, str] = {"limit": str(args.limit)}
    if args.task_type:
        params["task_type"] = args.task_type
    if args.status:
        params["status"] = args.status
    if args.search:
        params["q"] = args.search
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30) as c:
        r = c.get("/jobs", params=params)
        r.raise_for_status()
        rows = r.json().get("jobs", [])
    if not rows:
        print("(no jobs)")
        return
    for j in rows:
        cost = f"${j['cost_usd']}" if j.get("cost_usd") is not None else "-"
        lat = f"{j['latency_ms']}ms" if j.get("latency_ms") is not None else "-"
        print(
            f"{j['job_id']}  {j['status']:<9} {j['task_type']:<20} "
            f"{(j.get('model_used') or '-'):<16} {j['client_app']:<20} {lat:>8} {cost:>8}  "
            f"{j['created_at']}"
        )


def cmd_jobs_get(args: argparse.Namespace) -> int:
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30) as c:
        r = c.get(f"/jobs/{args.job_id}")
        if r.status_code == 404:
            print("job not found", file=sys.stderr)
            return 1
        r.raise_for_status()
        j = r.json()
    for k in (
        "job_id", "task_type", "status", "model_used", "tokens_in", "tokens_out",
        "cost_usd", "latency_ms", "created_at", "completed_at", "error",
    ):
        if j.get(k) is not None:
            print(f"{k:<13}: {j[k]}")

    scores = (j.get("metadata") or {}).get("quality_scores", [])
    if scores:
        vals = [s["score"] for s in scores if isinstance(s.get("score"), int | float)]
        avg = sum(vals) / len(vals) if vals else 0
        print(f"\n--- eval ({len(scores)} score(s), avg {avg:.1f}) ---")
        for s in scores:
            via = s.get("via") or "?"
            reviewer = s.get("reviewer") or "?"
            note = f"  {s['note']!r}" if s.get("note") else ""
            print(f"  {s['score']}/5  via={via:<5} by {reviewer:<16} {s.get('at', '')}{note}")

    if j.get("response"):
        print("\n--- response ---")
        print(j["response"])
    return 0


def cmd_routing_list(args: argparse.Namespace) -> None:
    """List routing rules (admin)."""
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30) as c:
        r = c.get("/routing")
        r.raise_for_status()
        rules = r.json().get("rules", [])
    if not rules:
        print("(no routing rules)")
        return
    width = max(len(r["task_type"]) for r in rules)
    for rule in rules:
        shadows = ", ".join(
            f"{s['model']}@{s['rate']}" for s in rule.get("eval_shadow_models", [])
        )
        line = (
            f"{rule['task_type']:<{width}}  {rule['preferred_model']:<16} -> "
            f"{rule['fallback_model']:<18} [{rule['sensitivity']}]"
        )
        if shadows:
            line += f"  shadows: {shadows}"
        print(line)


# --- argparse wiring ----


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="conduct",
        description="Admin CLI for a running Conduct instance.",
    )
    subs = p.add_subparsers(dest="resource", required=True)

    prompts = subs.add_parser("prompts", help="manage prompt library")
    psubs = prompts.add_subparsers(dest="action", required=True)

    plist = psubs.add_parser("list", help="list all prompts")
    plist.set_defaults(func=cmd_prompts_list)

    pget = psubs.add_parser("get", help="print a prompt's current content")
    pget.add_argument("task_type")
    pget.add_argument("--client", help="restrict to this client's override")
    pget.set_defaults(func=cmd_prompts_get)

    pedit = psubs.add_parser("edit", help="open $EDITOR on a prompt; save with PUT")
    pedit.add_argument("task_type")
    pedit.add_argument("--client", help="edit a per-client override")
    pedit.set_defaults(func=cmd_prompts_edit)

    phist = psubs.add_parser("history", help="show recent edits for a prompt")
    phist.add_argument("task_type")
    phist.add_argument("--client", help="restrict to this client's override")
    phist.add_argument("--limit", type=int, default=20, help="max rows (default 20)")
    phist.set_defaults(func=cmd_prompts_history)

    jobs = subs.add_parser("jobs", help="inspect jobs across all clients")
    jsubs = jobs.add_subparsers(dest="action", required=True)

    jlist = jsubs.add_parser("list", help="list recent jobs (newest first)")
    jlist.add_argument("--task-type", dest="task_type", help="filter by task_type")
    jlist.add_argument("--status", help="filter by status")
    jlist.add_argument("--search", help="prompt substring match")
    jlist.add_argument("--limit", type=int, default=50, help="max rows (default 50)")
    jlist.set_defaults(func=cmd_jobs_list)

    jget = jsubs.add_parser("get", help="show a job's status + result by id")
    jget.add_argument("job_id")
    jget.set_defaults(func=cmd_jobs_get)

    routing = subs.add_parser("routing", help="inspect routing rules")
    rsubs = routing.add_subparsers(dest="action", required=True)
    rlist = rsubs.add_parser("list", help="list routing rules")
    rlist.set_defaults(func=cmd_routing_list)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        # Subcommands either return an explicit exit code (e.g. `get` returns
        # 1 on 404) or return None for "success — exit 0". sys.exit() handles
        # None natively, but coerce here so the intent is explicit.
        rc = args.func(args)
        sys.exit(rc if isinstance(rc, int) else 0)
    except httpx.HTTPStatusError as e:
        # Render a useful message — the body usually has FastAPI's `detail`.
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            detail = e.response.text[:200]
        print(f"HTTP {e.response.status_code}: {detail}", file=sys.stderr)
        sys.exit(1)
    except httpx.RequestError as e:
        print(f"request failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
