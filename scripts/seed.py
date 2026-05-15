"""Idempotent bootstrap: seeds client apps and routing rules from yaml configs.

Reads:
  - config/seed.clients.yaml    (gitignored; copy from .example.yaml)
  - config/seed.routing.yaml    (committed defaults)

Both files are optional. Re-running is safe — existing rows are left untouched.
Routing rules are editable live via `PUT /routing/{task_type}` after seeding.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select

from auth import generate_api_key, hash_api_key
from db.session import SessionLocal, engine
from models.client import ClientApp
from models.routing import RoutingRule
from models.types import Sensitivity

REPO_ROOT = Path(__file__).resolve().parent.parent
CLIENTS_PATH = REPO_ROOT / "config" / "seed.clients.yaml"
ROUTING_PATH = REPO_ROOT / "config" / "seed.routing.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


async def _seed_clients(
    session, specs: list[dict]
) -> tuple[list[tuple[str, str]], list[str]]:
    """Idempotent client insertion. Returns (created [(name, raw_key)], skipped names)."""
    created: list[tuple[str, str]] = []
    skipped: list[str] = []
    for spec in specs:
        name = spec["name"]
        if await session.scalar(select(ClientApp).where(ClientApp.name == name)) is not None:
            skipped.append(name)
            continue
        raw_key = generate_api_key()
        session.add(
            ClientApp(
                name=name,
                api_key_hash=hash_api_key(raw_key),
                notes=spec.get("notes", ""),
                rate_limit_per_minute=spec.get("rate_limit_per_minute"),
                allow_cloud_for_internal=bool(spec.get("allow_cloud_for_internal", False)),
            )
        )
        created.append((name, raw_key))
    return created, skipped


async def _seed_routing(session, specs: list[dict]) -> tuple[list[str], list[str]]:
    """Idempotent routing-rule insertion. Returns (created task_types, skipped)."""
    created: list[str] = []
    skipped: list[str] = []
    for spec in specs:
        tt = spec["task_type"]
        if await session.scalar(select(RoutingRule).where(RoutingRule.task_type == tt)) is not None:
            skipped.append(tt)
            continue
        # Validate sensitivity early — surface bad config as a clear error.
        sensitivity = Sensitivity(spec.get("sensitivity", Sensitivity.INTERNAL.value))
        session.add(
            RoutingRule(
                task_type=tt,
                preferred_model=spec["preferred_model"],
                fallback_model=spec["fallback_model"],
                sensitivity=sensitivity.value,
                max_tokens=int(spec.get("max_tokens", 1000)),
                notes=spec.get("notes", ""),
            )
        )
        created.append(tt)
    return created, skipped


def _report(
    created_clients: list[tuple[str, str]],
    skipped_clients: list[str],
    created_rules: list[str],
    skipped_rules: list[str],
) -> None:
    if skipped_clients:
        print(
            f"clients already present (skipped): {', '.join(skipped_clients)}",
            file=sys.stderr,
        )
    if skipped_rules:
        print(
            f"routing rules already present (skipped): {', '.join(skipped_rules)}",
            file=sys.stderr,
        )
    if created_rules:
        print(f"\nCreated {len(created_rules)} routing rules: {', '.join(created_rules)}")
    if created_clients:
        print("\nCreated clients — store these keys somewhere safe, they will not be shown again:")
        for name, key in created_clients:
            print(f"  {name}: {key}")
    if not (created_rules or created_clients):
        print("nothing new to create.", file=sys.stderr)


async def seed() -> int:
    clients_doc = _load_yaml(CLIENTS_PATH)
    routing_doc = _load_yaml(ROUTING_PATH)
    seed_clients = clients_doc.get("clients") or []
    seed_routing = routing_doc.get("rules") or []

    if not seed_clients and not CLIENTS_PATH.is_file():
        print(
            f"note: {CLIENTS_PATH.relative_to(REPO_ROOT)} not found — "
            f"copy {CLIENTS_PATH.name.replace('.yaml', '.example.yaml')} to "
            f"start (no clients will be seeded for now)",
            file=sys.stderr,
        )

    try:
        async with SessionLocal() as session:
            created_clients, skipped_clients = await _seed_clients(session, seed_clients)
            created_rules, skipped_rules = await _seed_routing(session, seed_routing)
            await session.commit()
    finally:
        await engine.dispose()

    _report(created_clients, skipped_clients, created_rules, skipped_rules)
    return 0


def main() -> None:
    sys.exit(asyncio.run(seed()))


if __name__ == "__main__":
    main()
