# Prompts and the `conduct` CLI

System prompts live in **Postgres**, not the filesystem. Each task type has a
shared default and optional per-client overrides, edited live via the API or the
`conduct` CLI — no redeploy, hot-reloaded on the next job.

## Resolution order

For a job with `task_type=X` from client `Y`, the resolver picks:

1. `prompts` row where `task_type=X AND client_id=Y.id` — the client override, if present
2. `prompts` row where `task_type=X AND client_id IS NULL` — the shared default

If neither exists, the job fails at resolution time. (A client override doesn't
require a shared default to exist.)

## Versioning

Every save appends a row to `prompt_versions` (an append-only audit log), and
each job records the `version_id` it resolved to in `Job.metadata.prompt`. So a
job stays tied to the exact prompt content it ran against, even after the prompt
is later edited. This replaced the old git-SHA provenance.

## Seed material vs. source of truth

`prompts/shared/*.md` and `prompts/clients/<name>/*.md` are **seed material
only** — `scripts/seed.py` imports them once into the `prompts` table on first
run. After that, the **DB is the source of truth**; editing the `.md` files does
nothing until a fresh seed of an empty table. Do ongoing edits via the CLI/API.

(Per-client prompt files often live in a private overlay repo and are symlinked
into `prompts/clients/<name>/`; the directory name must match a seeded client
app's name.)

## The `conduct` CLI

Installed as a console script (`pyproject.toml` → `[project.scripts]`). It talks
to a running Conduct server.

**Environment:**

- `CONDUCT_ADMIN_KEY` — required (admin auth).
- `CONDUCT_BASE_URL` — defaults to `http://localhost:8000`.

**Commands:**

```bash
conduct prompts list                                  # all prompts (shared + overrides)
conduct prompts get  bio_generation                   # print the shared prompt
conduct prompts get  bio_generation --client acme     # print acme's override
conduct prompts edit bio_generation                   # open $EDITOR, save on write
conduct prompts edit bio_generation --client acme     # edit acme's override
conduct prompts history bio_generation [--limit N]    # recent edits (version log)
```

`edit` is the headline command — it round-trips through `$EDITOR` like
`git commit -e`: it fetches the current content into a tempfile, opens your
editor, and `PUT`s the result on save. An empty or unchanged buffer aborts with
no write. Editing a `task_type` that doesn't exist yet starts from a blank buffer
and creates it.

## The HTTP API

The CLI is a thin wrapper over these admin endpoints:

| Method | Path | Notes |
|---|---|---|
| GET | `/prompts` | list all (shared + per-client) |
| GET | `/prompts/{task_type}` | current content; `?client=<name>` for an override |
| PUT | `/prompts/{task_type}` | upsert content; appends a `prompt_versions` row |
| GET | `/prompts/{task_type}/history` | recent versions |

Browse the same data read-only in the UI under **Tasks** (`/ui/tasks`), which
shows each task's routing rule alongside its shared prompt and client overrides,
with a version-history drawer.

## Authoring guidelines

- Keep system prompts focused on **role + constraints + output shape** — not
  user data. The per-job `prompt` field carries the content.
- Prefer concrete examples over abstract instructions when behavior matters.
- A per-client override fully replaces the shared prompt for that client; it's
  not merged.
