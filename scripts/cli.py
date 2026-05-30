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
_CLIENTS_PATH = "/clients"
_CONNECTORS_PATH = "/connectors"
_CLIENT_ARG_HELP = "client name or UUID"
_CONNECTOR_ARG_HELP = "connector name or UUID"


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
    if args.min_score is not None:
        params["min_score"] = str(args.min_score)
    if args.max_score is not None:
        params["max_score"] = str(args.max_score)
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
        score = (
            f"{j['avg_score']:.1f}x{j['score_count']}" if j.get("avg_score") is not None else "-"
        )
        print(
            f"{j['job_id']}  {j['status']:<9} {j['task_type']:<20} "
            f"{(j.get('model_used') or '-'):<16} {j['client_app']:<20} {lat:>8} {cost:>8} "
            f"{score:>6}  {j['created_at']}"
        )


def _print_eval_scores(scores: list[dict]) -> None:
    if not scores:
        return
    vals = [s["score"] for s in scores if isinstance(s.get("score"), int | float)]
    avg = sum(vals) / len(vals) if vals else 0
    print(f"\n--- eval ({len(scores)} score(s), avg {avg:.1f}) ---")
    for s in scores:
        via = s.get("via") or "?"
        reviewer = s.get("reviewer") or "?"
        note = f"  {s['note']!r}" if s.get("note") else ""
        print(f"  {s['score']}/5  via={via:<5} by {reviewer:<16} {s.get('at', '')}{note}")


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

    _print_eval_scores((j.get("metadata") or {}).get("quality_scores", []))

    if j.get("response"):
        print("\n--- response ---")
        print(j["response"])
    return 0


def _resolve_client(c: httpx.Client, name_or_id: str) -> dict:
    """Look up a client by name or UUID via the admin list. The admin API uses
    UUIDs everywhere, but operators think in names, so the CLI does the
    translation. Exits with status 1 if not found."""
    r = c.get(_CLIENTS_PATH)
    r.raise_for_status()
    for row in r.json():
        if row["name"] == name_or_id or row["id"] == name_or_id:
            return row
    print(f"client {name_or_id!r} not found", file=sys.stderr)
    sys.exit(1)


def cmd_clients_list(args: argparse.Namespace) -> None:
    """List client apps."""
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30) as c:
        r = c.get(_CLIENTS_PATH)
        r.raise_for_status()
        rows = r.json()
    if not rows:
        print("(no clients)")
        return
    width = max(len(r["name"]) for r in rows)
    for r in rows:
        flags = []
        if not r["is_active"]:
            flags.append("inactive")
        if r["allow_cloud_for_internal"]:
            flags.append("cloud-for-internal")
        if r.get("rate_limit_per_minute") is not None:
            flags.append(f"rate:{r['rate_limit_per_minute']}/min")
        suffix = "  " + " ".join(f"[{f}]" for f in flags) if flags else ""
        print(f"  {r['name']:<{width}}  {r['id']}  key={r.get('key_created_at', '?')}{suffix}")


def _print_reveal_once(name: str, raw_key: str, action: str = "created") -> None:
    print(f"{action} client {name}")
    print(f"  api_key: {raw_key}")
    print("  (this is the only time the raw key will be shown — save it now)")


def cmd_clients_create(args: argparse.Namespace) -> None:
    body: dict = {"name": args.name, "notes": args.notes or ""}
    if args.rate_limit is not None:
        body["rate_limit_per_minute"] = args.rate_limit
    if args.allow_cloud_for_internal:
        body["allow_cloud_for_internal"] = True
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30) as c:
        r = c.post(_CLIENTS_PATH, json=body)
        r.raise_for_status()
        out = r.json()
    _print_reveal_once(out["name"], out["api_key"], "created")
    print(f"  id: {out['id']}")


def cmd_clients_rotate_key(args: argparse.Namespace) -> int:
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30) as c:
        client = _resolve_client(c, args.client)
        if not args.yes:
            answer = input(
                f"Rotate API key for {client['name']}? "
                "Old key stops working immediately. [y/N] "
            ).strip().lower()
            if answer != "y":
                print("aborted", file=sys.stderr)
                return 1
        r = c.post(f"/clients/{client['id']}/rotate-key")
        r.raise_for_status()
        out = r.json()
    _print_reveal_once(out["name"], out["api_key"], "rotated key for")
    return 0


def cmd_clients_toggle(args: argparse.Namespace) -> None:
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30) as c:
        client = _resolve_client(c, args.client)
        new_state = not client["is_active"]
        r = c.patch(f"/clients/{client['id']}", json={"is_active": new_state})
        r.raise_for_status()
    print(f"{client['name']} is now {'active' if new_state else 'inactive'}")


def cmd_clients_usage(args: argparse.Namespace) -> None:
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30) as c:
        client = _resolve_client(c, args.client)
        r = c.get(f"/clients/{client['id']}/usage", params={"days": str(args.days)})
        r.raise_for_status()
        out = r.json()
    print(f"{client['name']} usage over last {out['period_days']} days:")
    print(f"  jobs       : {out['job_count']}")
    print(f"  tokens_in  : {out['tokens_in']}")
    print(f"  tokens_out : {out['tokens_out']}")
    print(f"  cost_usd   : ${out['cost_usd']}")
    if out["by_day"]:
        print("  daily:")
        for d in out["by_day"]:
            print(
                f"    {d['date']}  jobs={d['job_count']:<4}"
                f"  in={d['tokens_in']:<6} out={d['tokens_out']:<6}  cost=${d['cost_usd']}"
            )


def _resolve_connector(c: httpx.Client, name_or_id: str) -> dict:
    r = c.get(_CONNECTORS_PATH)
    r.raise_for_status()
    for row in r.json():
        if row["name"] == name_or_id or row["id"] == name_or_id:
            return row
    print(f"connector {name_or_id!r} not found", file=sys.stderr)
    sys.exit(1)


def cmd_connectors_list(args: argparse.Namespace) -> None:
    """List OAuth connectors (MCP clients)."""
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30) as c:
        r = c.get(_CONNECTORS_PATH)
        r.raise_for_status()
        rows = r.json()
        # Build a client_app_id → name map for display.
        rc = c.get(_CLIENTS_PATH)
        rc.raise_for_status()
        client_names = {a["id"]: a["name"] for a in rc.json()}
    if not rows:
        print("(no connectors)")
        return
    width = max(len(r["name"]) for r in rows)
    for r in rows:
        flags = [] if r["is_active"] else ["inactive"]
        suffix = "  " + " ".join(f"[{f}]" for f in flags) if flags else ""
        bound = client_names.get(r["client_app_id"], r["client_app_id"])
        print(f"  {r['name']:<{width}}  {r['client_id']}  client={bound}{suffix}")


def _reveal_secret(name: str, client_id: str, client_secret: str, action: str) -> None:
    print(f"{action} connector {name}")
    print(f"  client_id    : {client_id}")
    print(f"  client_secret: {client_secret}")
    print("  (this is the only time the secret will be shown — save it now)")


def cmd_connectors_create(args: argparse.Namespace) -> None:
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30) as c:
        client = _resolve_client(c, args.client)
        body: dict = {"name": args.name, "client_app_id": client["id"]}
        if args.redirect_uri:
            body["redirect_uris"] = list(args.redirect_uri)
        r = c.post(_CONNECTORS_PATH, json=body)
        r.raise_for_status()
        out = r.json()
    _reveal_secret(out["name"], out["client_id"], out["client_secret"], "created")


def cmd_connectors_rotate_secret(args: argparse.Namespace) -> int:
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30) as c:
        conn = _resolve_connector(c, args.connector)
        if not args.yes:
            answer = input(
                f"Rotate secret for connector {conn['name']}? "
                "Existing tokens keep working but the old secret stops minting new ones. [y/N] "
            ).strip().lower()
            if answer != "y":
                print("aborted", file=sys.stderr)
                return 1
        r = c.post(f"{_CONNECTORS_PATH}/{conn['id']}/rotate-secret")
        r.raise_for_status()
        out = r.json()
    _reveal_secret(out["name"], out["client_id"], out["client_secret"], "rotated secret for")
    return 0


def cmd_connectors_toggle(args: argparse.Namespace) -> None:
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30) as c:
        conn = _resolve_connector(c, args.connector)
        new_state = not conn["is_active"]
        r = c.patch(f"{_CONNECTORS_PATH}/{conn['id']}", json={"is_active": new_state})
        r.raise_for_status()
    print(f"{conn['name']} is now {'active' if new_state else 'inactive'}")


def cmd_eval_compare(args: argparse.Namespace) -> None:
    """Per-model rollup for a task_type (jobs ∪ shadows over N days)."""
    params = {"task_type": args.task_type, "days": str(args.days)}
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30) as c:
        r = c.get("/eval/compare", params=params)
        r.raise_for_status()
        out = r.json()
    rows = out.get("models", [])
    if not rows:
        print(f"(no eval data for task_type={args.task_type!r} in last {args.days}d)")
        return
    print(f"task_type={args.task_type}  window={args.days}d")
    print(
        f"  {'model':<22} {'jobs':>5} {'fails':>5} {'fail%':>6} "
        f"{'p50_lat_ms':>10} {'avg_tok_out':>11} {'cost/job':>10} {'avg_score':>10}"
    )
    for m in rows:
        score = (
            f"{m['avg_score']:.2f}/{m['score_count']}"
            if m.get("avg_score") is not None
            else "-"
        )
        lat = f"{int(m['avg_latency_ms'])}" if m.get("avg_latency_ms") is not None else "-"
        tok = f"{int(m['avg_tokens_out'])}" if m.get("avg_tokens_out") is not None else "-"
        print(
            f"  {m['model']:<22} {m['job_count']:>5} {m['failure_count']:>5} "
            f"{m['failure_rate']*100:>5.1f}% {lat:>10} {tok:>11} "
            f"${m['cost_per_job_usd']:>9.4f} {score:>10}"
        )


def cmd_eval_review(args: argparse.Namespace) -> None:
    """List completed shadows that haven't been scored yet."""
    params: dict[str, str] = {"limit": str(args.limit)}
    if args.task_type:
        params["task_type"] = args.task_type
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30) as c:
        r = c.get("/eval/review", params=params)
        r.raise_for_status()
        out = r.json()
    items = out.get("items", [])
    if not items:
        print("(no unscored shadows)")
        return
    for it in items:
        print(f"  shadow_id={it['shadow_id']}")
        print(
            f"    parent={it['parent_job_id']}  task={it['task_type']}  "
            f"model={it['model']}  at={it['created_at']}"
        )
        print(f"    prompt:   {it['prompt'][:120]!r}")
        print(f"    response: {it['response'][:200]!r}")
        print()


def cmd_eval_score(args: argparse.Namespace) -> int:
    body: dict = {"score": args.score}
    if args.note:
        body["note"] = args.note
    if args.reviewer:
        body["reviewer"] = args.reviewer
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30) as c:
        r = c.post(f"/eval/jobs/{args.target_id}/score", json=body)
        if r.status_code == 404:
            print("no job or shadow with that id", file=sys.stderr)
            return 1
        r.raise_for_status()
        out = r.json()
    print(f"scored {out['kind']} {out['id']} -> total scores: {len(out['scores'])}")
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
    jlist.add_argument("--min-score", dest="min_score", type=float, help="min avg eval score (1-5)")
    jlist.add_argument("--max-score", dest="max_score", type=float, help="max avg eval score (1-5)")
    jlist.add_argument("--limit", type=int, default=50, help="max rows (default 50)")
    jlist.set_defaults(func=cmd_jobs_list)

    jget = jsubs.add_parser("get", help="show a job's status + result by id")
    jget.add_argument("job_id")
    jget.set_defaults(func=cmd_jobs_get)

    routing = subs.add_parser("routing", help="inspect routing rules")
    rsubs = routing.add_subparsers(dest="action", required=True)
    rlist = rsubs.add_parser("list", help="list routing rules")
    rlist.set_defaults(func=cmd_routing_list)

    clients = subs.add_parser("clients", help="manage client apps")
    csubs = clients.add_subparsers(dest="action", required=True)

    clist = csubs.add_parser("list", help="list client apps")
    clist.set_defaults(func=cmd_clients_list)

    ccreate = csubs.add_parser("create", help="create a client (raw key shown once)")
    ccreate.add_argument("name")
    ccreate.add_argument("--notes", default="", help="optional free-text notes")
    ccreate.add_argument(
        "--rate-limit", dest="rate_limit", type=int, help="requests per minute"
    )
    ccreate.add_argument(
        "--allow-cloud-for-internal",
        dest="allow_cloud_for_internal",
        action="store_true",
        help="permit cloud models on internal-sensitivity jobs",
    )
    ccreate.set_defaults(func=cmd_clients_create)

    crotate = csubs.add_parser(
        "rotate-key", help="mint a new API key for a client (old stops working)"
    )
    crotate.add_argument("client", help=_CLIENT_ARG_HELP)
    crotate.add_argument("-y", "--yes", action="store_true", help="skip confirmation")
    crotate.set_defaults(func=cmd_clients_rotate_key)

    ctoggle = csubs.add_parser("toggle", help="flip a client's active flag")
    ctoggle.add_argument("client", help=_CLIENT_ARG_HELP)
    ctoggle.set_defaults(func=cmd_clients_toggle)

    cusage = csubs.add_parser("usage", help="show usage stats for a client")
    cusage.add_argument("client", help=_CLIENT_ARG_HELP)
    cusage.add_argument("--days", type=int, default=30, help="lookback (default 30)")
    cusage.set_defaults(func=cmd_clients_usage)

    conns = subs.add_parser("connectors", help="manage MCP OAuth connectors")
    nsubs = conns.add_subparsers(dest="action", required=True)

    nlist = nsubs.add_parser("list", help="list OAuth connectors")
    nlist.set_defaults(func=cmd_connectors_list)

    ncreate = nsubs.add_parser(
        "create", help="create a connector (client_id + secret shown once)"
    )
    ncreate.add_argument("name", help="connector name (e.g. dave-ios)")
    ncreate.add_argument("--client", required=True, help=_CLIENT_ARG_HELP)
    ncreate.add_argument(
        "--redirect-uri",
        action="append",
        help="allowed redirect URI (repeatable; defaults to Claude's callback)",
    )
    ncreate.set_defaults(func=cmd_connectors_create)

    nrotate = nsubs.add_parser("rotate-secret", help="mint a new client secret")
    nrotate.add_argument("connector", help=_CONNECTOR_ARG_HELP)
    nrotate.add_argument("-y", "--yes", action="store_true", help="skip confirmation")
    nrotate.set_defaults(func=cmd_connectors_rotate_secret)

    ntoggle = nsubs.add_parser(
        "toggle", help="flip a connector's active flag (revokes its tokens if disabled)"
    )
    ntoggle.add_argument("connector", help=_CONNECTOR_ARG_HELP)
    ntoggle.set_defaults(func=cmd_connectors_toggle)

    ev = subs.add_parser("eval", help="model comparison + scoring")
    esubs = ev.add_subparsers(dest="action", required=True)

    ecmp = esubs.add_parser("compare", help="per-model rollup for a task_type")
    ecmp.add_argument("task_type")
    ecmp.add_argument("--days", type=int, default=30, help="lookback (default 30)")
    ecmp.set_defaults(func=cmd_eval_compare)

    erev = esubs.add_parser("review", help="list shadows that haven't been scored")
    erev.add_argument("--task-type", dest="task_type", help="filter by task_type")
    erev.add_argument("--limit", type=int, default=10, help="max items (default 10)")
    erev.set_defaults(func=cmd_eval_review)

    escore = esubs.add_parser("score", help="record a 1-5 quality score for a job or shadow")
    escore.add_argument("target_id", help="job_id or shadow_id (UUID)")
    escore.add_argument("score", type=int, choices=[1, 2, 3, 4, 5])
    escore.add_argument("--note", help="short rationale")
    escore.add_argument("--reviewer", help="reviewer label (defaults to '')")
    escore.set_defaults(func=cmd_eval_score)

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
