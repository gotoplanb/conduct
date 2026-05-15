# Client-specific prompt overrides — seed material

> **Heads up:** `prompts/` is now seed material, not the live source of truth.
> `scripts/seed.py` imports these files into the Postgres `prompts` table on
> first run. After that, the DB row wins — edits via the `conduct` CLI or
> `PUT /prompts/{task_type}` are what take effect at runtime.

This directory is **gitignored**. Drop your deployment-specific prompts here
for the seed step (or, once seeded, edit them in-place with `conduct prompts edit`).

## Structure

```
prompts/clients/
└── {client_name}/             — must match the ClientApp.name in your DB
    └── {task_type}.md         — seeds a client-specific override for this task
```

Resolution at runtime: when a job arrives, Conduct authenticates the
`ClientApp` from its bearer token, then queries the `prompts` table for a
row with `(task_type, client_id=<client.id>)`. If that row is missing, it
falls back to the shared row (`client_id IS NULL`).

## Two ways to manage these

**Option A — local-only files (simplest):**
Create files directly here, then `make seed`. Gitignored, never tracked.

**Option B — versioned via a separate repo (recommended for orgs):**
Maintain a private repo of overrides and mount it as a git submodule:

```bash
git submodule add git@github.com:your-org/conduct-prompts.git prompts/clients
git submodule update --init --recursive
```

The `prompts/clients/*` gitignore rule already accommodates this — Git
submodule contents won't be re-tracked by the parent repo.

## Example

`bio_generation.md` in this `.example/` folder shows the shape of a client
override. Copy it to `prompts/clients/{your-client-name}/bio_generation.md`
and fill in the bracketed details, then run `make seed`.

## Editing after seed

```bash
export CONDUCT_ADMIN_KEY=...
conduct prompts edit bio_generation --client {your-client-name}
```

This opens `$EDITOR` on the live content and PUTs the result on save. Every
save appends a row to `prompt_versions`, and each job records the
`version_id` it resolved to in `Job.metadata.prompt` — so you can always
trace back to the exact prompt content a given job ran against.
