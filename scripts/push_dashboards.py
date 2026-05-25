"""Push Conduct's Grafana dashboards into a Grafana instance, idempotently.

Dashboards live as JSON in `grafana/dashboards/` (versioned in this repo, not
the public watchtower one). Each JSON is the dashboard model with a stable
`uid`; this script POSTs them to `/api/dashboards/db` with overwrite=true, so
re-running updates in place rather than creating duplicates.

Auth (env): GRAFANA_URL (default http://localhost:3000) plus either
GRAFANA_TOKEN (a service-account token) or GRAFANA_USER/GRAFANA_PASSWORD.

Usage: `python -m scripts.push_dashboards` (or `make grafana-dashboards`).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "grafana" / "dashboards"


def _auth() -> dict:
    token = os.environ.get("GRAFANA_TOKEN", "").strip()
    if token:
        return {"headers": {"Authorization": f"Bearer {token}"}}
    user = os.environ.get("GRAFANA_USER", "admin")
    password = os.environ.get("GRAFANA_PASSWORD", "")
    if not password:
        print(
            "error: set GRAFANA_TOKEN, or GRAFANA_USER + GRAFANA_PASSWORD.", file=sys.stderr
        )
        sys.exit(2)
    return {"auth": (user, password)}


def _push_one(client: httpx.Client, path: Path) -> None:
    dashboard = json.loads(path.read_text(encoding="utf-8"))
    payload = {"dashboard": dashboard, "overwrite": True, "message": "pushed by conduct"}
    r = client.post("/api/dashboards/db", json=payload)
    r.raise_for_status()
    body = r.json()
    print(f"  {path.name} -> uid={body.get('uid')} (v{body.get('version')}) {body.get('url', '')}")


def main() -> int:
    base = os.environ.get("GRAFANA_URL", "http://localhost:3000").rstrip("/")
    files = sorted(DASHBOARD_DIR.glob("*.json"))
    if not files:
        print(f"no dashboards found in {DASHBOARD_DIR}", file=sys.stderr)
        return 1

    print(f"pushing {len(files)} dashboard(s) to {base}")
    with httpx.Client(base_url=base, timeout=30, **_auth()) as client:
        try:
            for path in files:
                _push_one(client, path)
        except httpx.HTTPStatusError as e:
            print(f"error: HTTP {e.response.status_code}: {e.response.text[:200]}", file=sys.stderr)
            return 1
        except httpx.RequestError as e:
            print(f"error: request failed: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
