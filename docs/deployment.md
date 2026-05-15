# Deployment

How Conduct is packaged for local dev, what's gitignored vs. committed, and what to keep in mind when targeting a cloud runtime.

## Local: containers vs. host-side

Two ways to run the API and worker locally:

- **`make up`** — full stack via Docker Compose (`postgres`, `redis`, `api`, `worker`). Lifecycle is managed by Compose; no orphan processes when you close your terminal. Use for prod-parity testing and as the default once your changes have settled.
- **`make up-infra` + `make run` + `make worker`** — only postgres + redis containerized; api and worker run on the host via uv. Use for fast iteration: uvicorn `--reload` rebuilds in ~100ms on file save.

Both paths share the same `.env` file. Compose overrides the localhost-flavored URLs for `DATABASE_URL`, `REDIS_URL`, `OLLAMA_BASE_URL`, and `OTEL_EXPORTER_OTLP_ENDPOINT` so containerized services reach the right hosts (`postgres:5432`, `redis:6379`, `host.docker.internal:11434`, `host.docker.internal:4317`).

**Ollama always stays on the host.** Docker Desktop on macOS can't pass through Metal/GPU access, which kills 70b-class model performance. Run `brew services start ollama` once and leave it.

## Container image

`Dockerfile` is multi-stage:

- **Builder** — `uv` 0.5 image; runs `uv sync --frozen --no-dev` to populate `/app/.venv` from `uv.lock`.
- **Runtime** — `python:3.12-slim-bookworm`; copies the venv and source, drops to a non-root `conduct` user (uid 1001), exposes 8000 + 8001.

The same image runs as either api or worker; Compose distinguishes them via the `command:` override. Built image is ~82MB content / ~389MB on disk.

### Prompt provenance

`metadata.prompt.version_id` records the `prompt_versions.id` a job resolved to — every edit through `PUT /prompts/{task_type}` (or `conduct prompts edit`) appends a row to `prompt_versions` before the live row is updated, so older jobs stay tied to their exact content even after later edits. Replay or audit by joining `Job.job_metadata->'prompt'->>'version_id'` against `prompt_versions.id`.

## Cloud target (ECS / Google Cloud Run)

Conduct is built with the assumption it'll eventually run on AWS ECS or Google Cloud Run. Implications already accounted for:

- API and worker ship as a single container image (one Dockerfile, two commands). ECS task definitions or Cloud Run services point at the same image with different entrypoints.
- The image runs as non-root with no host-volume requirements at runtime.
- All config flows through env vars (`DATABASE_URL`, `REDIS_URL`, `OLLAMA_BASE_URL`, `ANTHROPIC_API_KEY`, `CONDUCT_ADMIN_KEY`, `OTEL_EXPORTER_OTLP_ENDPOINT`).
- Migrations are a separate concern (`alembic upgrade head`), runnable as a one-off ECS task or a Cloud Run job before deploying the new revision.
- Prompts live in Postgres, edited via the admin API / CLI — no filesystem hot-reload needed, audit history travels with the database.

`seed.clients.yaml` is still a filesystem artifact for first-time bootstrap. In a cloud deploy that's a one-off concern: run `scripts/seed.py` once as part of the initial migration job (it's idempotent — safe to re-run), or mount the file as a secret and let the cold-start seed call pick it up.

## Private configuration (deployment-specific overrides)

Two paths are deployment-specific and **must not be committed to a public fork**:

| Path | Status | Purpose |
|---|---|---|
| `config/seed.clients.yaml` | gitignored | Client app names + per-client knobs (rate limits, cloud opt-in) |
| `prompts/clients/{client_name}/` | gitignored | Per-deployment override files — only used by `scripts/seed.py` on initial import. After seed, edits happen via the `conduct` CLI; the DB row is authoritative. |

`config/seed.clients.example.yaml` and `prompts/clients/.example/` ship as templates. Two patterns for managing the real values on a developer machine:

- **Local files** (simplest): copy the examples into the gitignored paths and fill in. Works for solo / single-Mac deployments.
- **Private repo** (recommended for orgs): keep your overrides in a separate private repo and either symlink them in or mount as a git submodule:
  ```bash
  git submodule add git@github.com:your-org/conduct-prompts.git prompts/clients
  git submodule update --init --recursive
  ```
  The gitignore rules already accommodate submodule contents — they won't be re-tracked by this repo.

`config/seed.routing.yaml` *is* committed because the starter rules are generic (task_types and model identifiers, no client identity). Override per-deployment by editing live (`PUT /routing/{task_type}`).

## ngrok

For exposing the local API over HTTPS to external services (e.g. a GCP function calling Conduct):

```bash
echo "NGROK_AUTHTOKEN=..." >> .env
ngrok config add-authtoken "$NGROK_AUTHTOKEN"
make tunnel              # starts `ngrok http 8000`
```

Copy the printed HTTPS URL into the caller's config. The tunnel is the only HTTPS surface; the underlying app speaks HTTP locally.
