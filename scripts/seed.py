"""Idempotent bootstrap: seeds client apps, routing rules, and prompts.

Reads:
  - config/seed.clients.yaml    (gitignored; copy from .example.yaml)
  - config/seed.routing.yaml    (committed defaults)
  - prompts/shared/*.md         (committed defaults — imported as shared prompts)
  - prompts/clients/<name>/*.md (gitignored or symlinked — imported as
                                 per-client overrides; <name> must match a
                                 seeded ClientApp.name)

All steps are idempotent. Re-running is safe — existing rows are left untouched.
After seeding, routing rules edit via `PUT /routing/{task_type}` and prompts via
`PUT /prompts/{task_type}` (or the `conduct prompts edit` CLI).
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
from models.prompt import Prompt, PromptVersion
from models.routing import RoutingRule
from models.types import MediaKind, Sampling, Sensitivity

REPO_ROOT = Path(__file__).resolve().parent.parent
CLIENTS_PATH = REPO_ROOT / "config" / "seed.clients.yaml"
ROUTING_PATH = REPO_ROOT / "config" / "seed.routing.yaml"
PROMPTS_DIR = REPO_ROOT / "prompts"


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
        # media_kind defaults to 'text' — keeps existing seed entries that
        # predate this column working unchanged.
        media_kind = MediaKind(spec.get("media_kind", MediaKind.TEXT.value))
        # sampling defaults to 'balanced' — same backward-compat rationale.
        sampling = Sampling(spec.get("sampling", Sampling.BALANCED.value))
        session.add(
            RoutingRule(
                task_type=tt,
                preferred_model=spec["preferred_model"],
                fallback_model=spec["fallback_model"],
                sensitivity=sensitivity.value,
                max_tokens=int(spec.get("max_tokens", 1000)),
                notes=spec.get("notes", ""),
                eval_shadow_models=spec.get("eval_shadow_models") or [],
                media_kind=media_kind.value,
                sampling=sampling.value,
            )
        )
        created.append(tt)
    return created, skipped


def _is_prompt_file(path: Path) -> bool:
    """Treat README.md and dotfiles as documentation, not prompts."""
    return path.suffix == ".md" and path.name != "README.md" and not path.name.startswith(".")


async def _import_prompt_file(
    session, file: Path, *, client_id, label: str
) -> str | None:
    """Idempotent single-file import. Returns the created label, or None
    if the row already existed."""
    tt = file.stem
    stmt = select(Prompt).where(Prompt.task_type == tt)
    if client_id is None:
        stmt = stmt.where(Prompt.client_id.is_(None))
    else:
        stmt = stmt.where(Prompt.client_id == client_id)
    if await session.scalar(stmt) is not None:
        return None
    content = file.read_text(encoding="utf-8")
    session.add(Prompt(task_type=tt, client_id=client_id, content=content, updated_by="seed"))
    session.add(
        PromptVersion(task_type=tt, client_id=client_id, content=content, edited_by="seed")
    )
    return f"{label}:{tt}"


async def _seed_shared_prompts(session) -> tuple[list[str], list[str]]:
    """Import prompts/shared/<task>.md as `client_id=NULL` rows."""
    created: list[str] = []
    skipped: list[str] = []
    shared_dir = PROMPTS_DIR / "shared"
    files = (
        sorted(p for p in shared_dir.iterdir() if _is_prompt_file(p))
        if shared_dir.is_dir()
        else []
    )
    for f in files:
        result = await _import_prompt_file(session, f, client_id=None, label="shared")
        (created if result else skipped).append(result or f"shared:{f.stem}")
    return created, skipped


async def _seed_client_prompts(session) -> tuple[list[str], list[str], list[str]]:
    """Import prompts/clients/<name>/<task>.md as per-client overrides.
    Folders with no matching ClientApp surface as warnings."""
    created: list[str] = []
    skipped: list[str] = []
    warnings: list[str] = []
    clients_root = PROMPTS_DIR / "clients"
    if not clients_root.is_dir():
        return created, skipped, warnings

    client_dirs = sorted(
        p for p in clients_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    for client_dir in client_dirs:
        client = await session.scalar(
            select(ClientApp).where(ClientApp.name == client_dir.name)
        )
        if client is None:
            warnings.append(
                f"prompts/clients/{client_dir.name}/ has no matching ClientApp — skipped"
            )
            continue
        for f in sorted(p for p in client_dir.iterdir() if _is_prompt_file(p)):
            result = await _import_prompt_file(
                session, f, client_id=client.id, label=client_dir.name
            )
            (created if result else skipped).append(result or f"{client_dir.name}:{f.stem}")
    return created, skipped, warnings


async def _seed_prompts(session) -> tuple[list[str], list[str], list[str]]:
    """Import `.md` files from prompts/shared and prompts/clients/<name>.

    Idempotent: a row that already exists is left alone. Newly-imported
    rows get a companion `prompt_versions` entry tagged `seed` so the
    audit log starts from a known point.
    """
    if not PROMPTS_DIR.is_dir():
        return [], [], []
    shared_created, shared_skipped = await _seed_shared_prompts(session)
    client_created, client_skipped, warnings = await _seed_client_prompts(session)
    return shared_created + client_created, shared_skipped + client_skipped, warnings


def _report(
    created_clients: list[tuple[str, str]],
    skipped_clients: list[str],
    created_rules: list[str],
    skipped_rules: list[str],
    created_prompts: list[str],
    skipped_prompts: list[str],
    prompt_warnings: list[str],
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
    if skipped_prompts:
        print(
            f"prompts already present (skipped): {', '.join(skipped_prompts)}",
            file=sys.stderr,
        )
    for w in prompt_warnings:
        print(f"warning: {w}", file=sys.stderr)
    if created_rules:
        print(f"\nCreated {len(created_rules)} routing rules: {', '.join(created_rules)}")
    if created_prompts:
        print(f"\nCreated {len(created_prompts)} prompts: {', '.join(created_prompts)}")
    if created_clients:
        print("\nCreated clients — store these keys somewhere safe, they will not be shown again:")
        for name, key in created_clients:
            print(f"  {name}: {key}")
    if not (created_rules or created_clients or created_prompts):
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
            # Clients must be visible to the prompt-importer (it joins by name),
            # so flush before scanning prompts/clients/<name>/.
            await session.flush()
            created_prompts, skipped_prompts, prompt_warnings = await _seed_prompts(session)
            await session.commit()
    finally:
        await engine.dispose()

    _report(
        created_clients,
        skipped_clients,
        created_rules,
        skipped_rules,
        created_prompts,
        skipped_prompts,
        prompt_warnings,
    )
    return 0


def main() -> None:
    sys.exit(asyncio.run(seed()))


if __name__ == "__main__":
    main()
