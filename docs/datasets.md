# Dataset export — training data from scored traffic

Conduct accumulates **scored comparisons** as a byproduct of being used: every
job + shadow can carry quality scores and pairwise preferences (mostly produced
by the [LLM-as-a-judge](judging.md), some by humans/clients — see [eval.md](eval.md)).
The dataset export turns that into the two shapes fine-tuning wants, as
streamed JSONL ready for TRL / unsloth:

| endpoint | shape | for |
|----------|-------|-----|
| `GET /datasets/sft` | `{prompt, system, completion, meta}` | supervised fine-tuning |
| `GET /datasets/preferences` | `{prompt, system, chosen, rejected, meta}` | DPO / preference tuning |

Both are **admin-only** and stream `application/x-ndjson` (one JSON object per
line). The flywheel: *judge scores accumulate → export → fine-tune → better
models → more scores*, with no human labelling in the loop.

> Conduct stays **task-agnostic**: it exports `prompt` / `response` / `score`
> and the lineage that makes them comparable. Domain-specific reshaping or
> enrichment is the tenant's job at export time (e.g. join `meta.id` back to
> your own store).

## SFT — `GET /datasets/sft`

High-scored responses become `(prompt, system, completion)` examples. A row
qualifies if the **average** of its `quality_scores` (optionally filtered to one
source) is `>= min_score`.

| param | default | meaning |
|-------|---------|---------|
| `task_type` | — | restrict to one task type |
| `min_score` | `4` | minimum average quality score (1–5) |
| `via` | — | only count scores from one source: `judge` / `judge-panel` / `admin` / `mcp` / `url` |
| `include_shadows` | `false` | also export high-scored **shadow** responses |
| `prompt_version` | — | restrict to one resolved prompt version (comparability) |
| `limit` | `1000` | max rows (cap 10000) |

```bash
curl -s "$CONDUCT/datasets/sft?min_score=4&via=judge&include_shadows=true" \
  -H "Authorization: Bearer $ADMIN_KEY"
```

```jsonl
{"prompt":"In one sentence, why is the sky blue?","system":"You are a precise science tutor…",
 "completion":"The sky appears blue because of Rayleigh scattering…",
 "meta":{"task_type":"qa","model":"gemma4:e4b","score":5.0,"n_scores":1,
         "prompt_version":7,"sensitivity":"internal","source":"job","id":"…"}}
```

The `system` field is reconstructed even for library-sourced jobs (it resolves
the recorded prompt version's content), so examples are usable as-is.

## Preferences (DPO) — `GET /datasets/preferences`

`(prompt, system, chosen, rejected)` pairs, by one of two methods:

- **`method=pairwise`** (default) — read the [pairwise judge](judging.md#pairwise-order-swapped)'s
  verdicts directly. Each order-swap-verified `win` becomes one pair (the mirror
  `loss` record is ignored, so no double-counting). The cleanest signal.
- **`method=score`** — derive pairs from pointwise/panel **score differentials**
  on the *same input*: for each parent job, take its own response + its shadows
  (all answers to one prompt), and pair the highest- vs lowest-scored when the
  gap `>= min_gap`. Comparable by construction.

| param | default | meaning |
|-------|---------|---------|
| `task_type` | — | restrict to one task type |
| `method` | `pairwise` | `pairwise` or `score` |
| `min_gap` | `2` | (score method) minimum score gap to form a pair |
| `prompt_version` | — | restrict to one prompt version |
| `limit` | `1000` | max rows (cap 10000) |

```bash
curl -s "$CONDUCT/datasets/preferences?method=pairwise&task_type=judge" \
  -H "Authorization: Bearer $ADMIN_KEY"
```

```jsonl
{"prompt":"In one sentence, why is the sky blue?","system":"…",
 "chosen":"The sky appears blue because of Rayleigh scattering…",
 "rejected":"It's because of, like, atmospheric conditions and stuff…",
 "meta":{"task_type":"qa","chosen_model":"…","rejected_model":"…",
         "prompt_version":7,"sensitivity":"internal","method":"pairwise","judge_job_id":"…"}}
```

## Comparability — don't mix apples and oranges

Two scored responses are only comparable if they answered under the same
conditions. The export keeps you honest:

- **Same input.** Both preference methods only pair responses to the *same*
  prompt (a parent + its shadows, or the two sides the judge compared) — never
  unrelated jobs.
- **Prompt version.** `meta.prompt_version` is on every row, and
  `?prompt_version=N` filters to one rubric/prompt revision — important when
  you've edited a prompt (scores from before vs after aren't comparable).
- **Score source.** `?via=judge` (or `judge-panel`) restricts to model-judged
  scores; useful when you want a clean automated signal without mixing in
  human/admin scores.

`meta.sensitivity` is on every row too — review before exporting
`confidential` content into a downloadable file.

## Consuming it

The output is line-delimited JSON. Save and feed to your trainer:

```bash
curl -s "$CONDUCT/datasets/preferences?method=pairwise" -H "Authorization: Bearer $ADMIN_KEY" \
  -o prefs.jsonl
# prefs.jsonl: one {prompt, system, chosen, rejected} per line → TRL DPOTrainer / unsloth
```

## Where the data comes from

- **Scores** are appended to `quality_scores` (1–5) by the judge
  (`via="judge"`/`"judge-panel"`), humans (review UI), or clients (`submit_eval`).
  See [eval.md](eval.md) and [judging.md](judging.md).
- **Pairwise preferences** are appended to `pairwise_verdicts` by the pairwise
  judge. See [judging.md](judging.md#pairwise-order-swapped).
- Shaping lives in `eval/datasets.py`; the streaming handlers in
  `routes/datasets.py`.

## Notes

- Admin auth required (these are bulk data pulls).
- Each export scans a bounded window of recent rows (a runaway guard); raise
  `limit` to pull more, narrow with `task_type` to focus.
- Multi-dimensional / named scores (e.g. separate `correctness` vs `format`) are
  not exported yet — today a row's score is the average of its `quality_scores`.
