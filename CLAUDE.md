# Conduct — Claude Code guidance

## Commit & push workflow

Once a change is **green on SonarQube** — quality gate `OK` (new_coverage ≥ 50,
new_violations = 0, new_duplicated_lines_density ≤ 3) **and** the test suite +
lint pass — you are authorized to commit and push to `origin/main` without
stopping to ask first.

The standing bar before commit/push:

1. `make sonar-scan` gate returns `OK` (check via
   `GET /api/qualitygates/project_status?projectKey=conduct`)
2. `uv run pytest` passes
3. `uv run ruff check` is clean

If any of those fail, fix the root cause and re-run — do not push red.
