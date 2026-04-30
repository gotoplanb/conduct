# Architecture

How Conduct decides where a job runs, when it runs synchronously vs. asynchronously, and what happens when the chosen model fails.

## Sensitivity tiers

Every routing rule and every job carries a `sensitivity`. The rule's value acts as a **floor** — clients can be stricter than the rule but never looser.

| tier            | routing                                                                       |
|-----------------|-------------------------------------------------------------------------------|
| `public`        | any model (local or cloud)                                                    |
| `internal`      | local preferred; cloud only when `ClientApp.allow_cloud_for_internal=true`    |
| `confidential`  | local only — hard gate, no cloud fallback ever, regardless of client opt-in   |

A `confidential` rule means cloud fallback is stripped from the decision before the request even reaches a provider, so a misconfigured client can't accidentally route confidential data to Anthropic.

## Sync vs async decision

`POST /jobs` picks one of two paths per request:

- **`"async": true` in body** → enqueue, return `202 Accepted` with a poll URL.
- **Cloud target** (Anthropic) → execute synchronously, return `200 OK` with the full result.
- **Local target** (Ollama) → enqueue. The worker is the sole owner of Ollama and model swaps, so synchronous local execution would force the API to coordinate model lifecycle from a request handler — which we don't want.

The same `worker/executor.py` runs both legs, so behavior is identical regardless of which path the API took.

## Routing engine

`routing/engine.py` is a pure function: `decide(rule, sensitivity_floor, available_models) -> Decision`. It:

1. Enforces the sensitivity floor (strips cloud options when `confidential`).
2. Picks `preferred_model` if it's loadable; falls back to `fallback_model` if not.
3. Returns the chosen model + provider + a `reason` string for audit (`rule:bio_generation`, `fallback:claude-sonnet-4-5`, etc.).

Rules live in Postgres (`routing_rules` table) and can be hot-reloaded via `PUT /routing/{task_type}` — no restart, no redeploy. Rules are read per request, so the next job after the PUT picks up the change.

## Failure handling

`retry/static.py` is the v1 `FailureHandler`:

- On a `ProviderError`, returns `FALLBACK` if a *distinct* fallback model+provider is loaded, else `FAIL`.
- "Distinct" means the fallback isn't the same model that just errored — a fallback to yourself is pointless.

`retry/triage.py` is the v2 stub: implementing its `on_provider_error()` and binding it as the default in `worker/executor.py` is the only change needed to swap from heuristics to an LLM-driven decision (e.g. "this looks like a transient rate limit, retry on the same model" vs "this looks like a context-length issue, route to a larger model").

Tenacity handles Anthropic rate-limit retries separately at the provider layer: 3 attempts, exponential backoff 1–10s. The `FailureHandler` only sees errors that survive that retry layer.

## Worker ownership

Why the worker — not the API — owns Ollama:

- Loading/unloading a 70b model takes 3–10s. If the API did it, every request handler would block waiting for swaps, and concurrent requests for different models would race.
- The worker is single-threaded (RQ `SimpleWorker`), so model swaps are naturally serialized. The "current loaded model" is implicit in the worker's last action.
- This means **all local-model jobs go async**, even ones the user might want sync. That's a design trade-off — single-Mac homelab deployment makes the queue overhead negligible.
