# Client-specific prompt overrides

This directory is **gitignored**. Drop your deployment-specific prompts here.

## Structure

```
prompts/clients/
└── {client_name}/             — must match the ClientApp.name in your DB
    └── {task_type}.md         — overrides prompts/shared/{task_type}.md for this client
```

`{client_name}` resolution: when a job arrives, Conduct authenticates the
`ClientApp` from its bearer token, then looks up
`prompts/clients/{ClientApp.name}/{task_type}.md`. If that file is missing, it
falls back to `prompts/shared/{task_type}.md`.

## Two ways to manage these

**Option A — local-only files (simplest):**
Just create files directly here. Gitignored, never tracked.

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
and fill in the bracketed details.

## Hot reload

Prompts are read from disk on every request. Save the file → next job uses
the new version. The git commit hash of the prompt file is captured per
job in `Job.metadata.prompt.git_hash` for auditability.
