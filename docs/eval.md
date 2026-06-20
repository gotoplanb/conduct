# Eval — comparing models for a task

Conduct's eval loop helps you answer "which model is best for this task?" using
**real production traffic** — measuring not just quality but cost and latency,
so you can pick the cheapest/fastest model that's good enough.

> **Related:** [judging.md](judging.md) automates the scoring (LLM-as-a-judge);
> [datasets.md](datasets.md) exports the scored data as SFT/DPO training JSONL.

## How it works

```
real job ──▶ runs on the production model (recorded)
   │
   └──▶ shadow(s): replay the same prompt against candidate model(s)
              on a low-priority queue, response not returned to the client
                     │
                     ▼
        rollup: per-model cost / latency / failure-rate / avg score
                     │
        review: rate the outputs 1–5 (human, in the UI)
```

1. **Shadows.** A routing rule can carry `eval_shadow_models` — candidate models
   that get replayed on a *sample* of real jobs. Shadows run on a separate
   low-priority queue, never block real traffic, and their responses aren't
   returned to the client. Each records the candidate's response, latency,
   tokens, and cost.
2. **Rollup.** For a task type over N days, Conduct aggregates per model: job
   count, failure rate, avg latency, avg tokens out, **cost per job**, and **avg
   quality score**. The production model (from real jobs) and the shadows sit in
   one table.
3. **Review.** Completed shadows are scored 1–5 by a human in the UI; scores
   feed the rollup's avg-score column.

> Quality scoring is human, by design. There's no automated judge yet;
> LLM-as-judge scoring is a possible future addition that would write into the
> same score slot the rollup already reads.

## Configuring shadows on a rule

`eval_shadow_models` is a list of `{ model, rate, daily_cost_cap_usd? }`:

- `model` — the candidate to shadow.
- `rate` — sampling fraction `0.0`–`1.0` (e.g. `1.0` = shadow every job; lower
  it once you've gathered enough data).
- `daily_cost_cap_usd` — optional, **cloud only**; stops shadowing that model
  once the day's spend hits the cap.

**Gating:** cloud shadows are skipped on `internal`/`confidential` jobs unless
the client has `allow_cloud_for_internal`. **Local models always run** (no cost,
no egress) — which is why a local-vs-local comparison needs no special flags.

### Set it live (hot-reloads)

```bash
curl -X PUT -H "Authorization: Bearer $CONDUCT_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  "$CONDUCT_BASE_URL/routing/bio_generation" \
  -d '{
    "preferred_model": "llama3.3:70b",
    "fallback_model": "claude-sonnet-4-5",
    "sensitivity": "internal",
    "eval_shadow_models": [{"model": "llama3.2:3b", "rate": 1.0}]
  }'
```

### Persist it across re-seeds

Add it to `config/seed.routing.yaml` so a fresh DB rebuild recreates it:

```yaml
  - task_type: bio_generation
    preferred_model: llama3.3:70b
    fallback_model: claude-sonnet-4-5
    sensitivity: internal
    eval_shadow_models:
      - model: llama3.2:3b
        rate: 1.0
```

`make seed` is idempotent — it won't overwrite an existing rule, so live edits
and the YAML can drift; treat the live DB as source of truth.

## Reviewing and comparing

- **Score outputs:** UI → **Eval → Review & score** (`/ui/eval/review`). Pick a
  task type; you see each job's prompt with the production answer and every
  shadow answer side by side, and rate each 1–5 inline.
- **Compare models:** UI → **Eval** (`/ui/eval`), or the JSON API:
  ```bash
  curl -H "Authorization: Bearer $CONDUCT_ADMIN_KEY" \
    "$CONDUCT_BASE_URL/eval/compare?task_type=bio_generation&days=30"
  ```
  Returns per-model `cost_per_job_usd`, `avg_latency_ms`, `failure_rate`,
  `avg_score`, and counts.

## Client-submitted scores

Beyond operator scoring, clients can submit their own quality scores. All paths
append to the same `quality_scores` list (so they feed the same rollup), and
each score is tagged with a `via` provenance field (`admin` / `mcp` / `url` /
`judge` / `judge-panel`) so you can filter operator vs. client- vs.
model-sourced scores later.

> **Automating the scoring itself?** The **`judge` task type** lets Conduct
> score (or compare, or jury) outputs through its own dispatch — pointwise,
> pairwise, and panel/jury modes, all feeding this same lane. See
> [judging.md](judging.md).

**From a Claude connector (MCP).** The `submit_eval` tool scores one of the
caller's own jobs. The calling Claude session interprets freeform feedback
("that joke was hilarious") into a 1–5 score *before* invoking the tool — no
extra round-trip or job:

```
Human: "that was great"
Claude → submit_eval(job_id, score=5, note="positive reaction")   # via=mcp
```

**From a credential-less rater (link).** For a portal/email scorer with no API
key, the owning client (or admin) mints a single-use token, then hands out the
link:

```bash
# 1. owner/admin mints a token
curl -X POST -H "Authorization: Bearer <key>" "$CONDUCT_BASE_URL/jobs/$JOB/eval-link"
#    → { "eval_url": ".../jobs/<id>/eval", "eval_token": "cdt_ev_…", "expires_at": ... }

# 2. the rater submits a score with just the token (no bearer auth)
curl -X POST "$CONDUCT_BASE_URL/jobs/$JOB/eval" \
  -d '{"eval_token": "cdt_ev_…", "score": 4, "note": "good but verbose"}'   # via=url
```

The token is single-use, TTL-bound (`EVAL_TOKEN_TTL_DAYS`, default 7),
job-scoped, and stored hashed. A second submit returns `409`; an expired or
wrong token returns `401`; a score outside 1–5 returns `422`.

> If Conduct sits behind an auth proxy (e.g. the ngrok `Bearer cdt_` edge
> guard), add `/jobs/{id}/eval` to the allowlist — the token-submit call comes
> from a credential-less browser and won't carry that header.

## Worked example: is a small local model good enough?

Goal: can `llama3.2:3b` replace `llama3.3:70b` for `bio_generation`?

1. Add `llama3.2:3b` as a shadow at `rate: 1.0` (above). Production stays 70b.
2. Let real `bio_generation` jobs flow; shadows accumulate automatically.
3. Score a batch of both models' outputs in the review UI.
4. Open the Eval rollup: if 3b's `avg_score` is acceptable at a fraction of the
   latency, you've found your model — switch the rule's `preferred_model`. If
   not, you have the data to justify staying on 70b.

The point is the *combined* picture: quality **and** cost/latency, on your real
prompts.

## Registering a fine-tuned model

A locally fine-tuned checkpoint (e.g. a DPO-tuned model from the code-gen
flywheel, [#22](https://github.com/gotoplanb/conduct/issues/22)) is a **drop-in
routing target** — there's no model registry to update. Conduct classifies any
name that isn't a `claude-*` (Anthropic) or dotted Bedrock id as a **local
Ollama** model: free pricing, dispatched via the worker's swap path.

1. Register the checkpoint in Ollama under any tag, e.g. `code-gen-dpo:v2`
   (`ollama create code-gen-dpo:v2 -f Modelfile`).
2. Point a rule at it — as `preferred_model`, a `fallback_model`, an
   `eval_shadow_models` entry, or a judge panel member — live via
   `PUT /routing/<task_type>` or persisted in `config/seed.routing.yaml`.
3. Its jobs + shadows roll up as their **own row** in `/eval/compare` (keyed on
   `model_used`), so you can compare the fine-tune head-to-head against the base
   model it improves — including the deterministic `composite` (#30).

The cleanest A/B: add the fine-tune as a `rate: 1.0` shadow of its base model
(above), let traffic flow, and read the rollup — same recipe as the worked
example, now measuring *your* model against its parent.

> **Naming caveat:** don't name a local checkpoint `claude-*` or with a Bedrock
> vendor prefix (`anthropic.…`, `meta.…`) — it'd be misrouted to a cloud
> provider. Any other tag is treated as local.
