# Conduct prompt library — seed material

Each `.md` file here is **seed content** for one task type. Filename is the
`task_type`. On first run, `scripts/seed.py` imports each file into the
Postgres `prompts` table; the DB is the source of truth after that.

## Resolution order (at job time)

For a job with `task_type=X` from `client_app=Y`:

1. `prompts` row where `task_type=X AND client_id=<Y.id>` — if present
2. `prompts` row where `task_type=X AND client_id IS NULL` (shared default) — fallback

The resolver fails the job if neither row exists.

## Editing after seed

The `.md` files in this directory are only consulted on initial import. For
ongoing edits, use the admin API or CLI:

```bash
conduct prompts edit bio_generation                          # shared
conduct prompts edit bio_generation --client bosshardt-portal  # override
conduct prompts history bio_generation                       # version log
```

Every save appends a row to `prompt_versions`; each job records the
`version_id` it resolved to in `Job.metadata.prompt`, so jobs stay tied to
the exact content they ran against even after later edits.

## Authoring guidelines

- Keep system prompts focused on **role + constraints + output shape**, not user data
- The user-supplied `prompt` field provides per-job content; don't try to anticipate it here
- Prefer concrete examples over abstract instructions when behavior matters
