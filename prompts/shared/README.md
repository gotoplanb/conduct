# Conduct prompt library

Each `.md` file is a system prompt for one task type. Filename is the `task_type`.

## Resolution order

For a job with `task_type=X` from `client_app=Y`:

1. `prompts/clients/{Y}/{X}.md` (client-specific override) — if present
2. `prompts/shared/{X}.md` (default) — fallback

The resolver fails the job if neither exists.

## Authoring guidelines

- Keep system prompts focused on **role + constraints + output shape**, not user data
- The user-supplied `prompt` field provides per-job content; don't try to anticipate it here
- Prefer concrete examples over abstract instructions when behavior matters
- Edits hot-reload — no restart needed
- Every job logs the resolved file path + the git commit SHA that last touched it
