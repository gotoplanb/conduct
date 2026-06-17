# LLM-as-a-judge — automated evaluation

Conduct already *consumes* judgements everywhere — shadow scores, the per-model
rollup ([eval.md](eval.md)), the planned DPO/SFT export ([#16](https://github.com/gotoplanb/conduct/issues/16)).
But by default those scores come from a **human** (the review UI, an eval-token
link) or an **external client**. The **`judge` task type** lets Conduct produce
them itself: a job whose work is to evaluate *another job's* output, routed
through Conduct's own dispatch.

It's pleasingly recursive — Conduct judging its own outputs through its own
engine — and most of the hard parts already existed (routing, deterministic
sampling, the score lane, fan-out, prompt versioning), so the judge mostly wires
primitives together. Design + research notes live in [#17](https://github.com/gotoplanb/conduct/issues/17).

## The one principle: a judge is a primitive, not a hook

Conduct's stance is **primitives, clients orchestrate, no lifecycle hooks**.
A judge is therefore an ordinary task you *submit* — it does not fire
automatically when jobs complete. It produces a verdict; **writing that verdict
back to the judged job happens only on an explicit `apply_to_target` opt-in.**
This keeps determinism guarantees and write-backs under the caller's control,
the same way media chaining works (you pass job-ids; Conduct resolves them).

A judge job is just a normal text job whose `inputs` name a target to evaluate:

```jsonc
{ "task_type": "judge", "prompt": "(judge)", "inputs": { "mode": "...", "target_job_id": "<uuid>", ... } }
```

Carrying a `target_job_id` is what marks it a judge (it then runs the judge
executor instead of a plain completion, and is forced onto the async worker).

## The three modes

| mode | compares | emits | feeds |
|------|----------|-------|-------|
| **pointwise** (default) | one target vs a rubric | `{score 1-5, rationale}` | monitoring, SFT labels |
| **pairwise** | two targets, head-to-head | `{chosen, rejected, tie}` | DPO (chosen, rejected) pairs |
| **panel** | one target, jury of models | `{score (median), n, disagreement}` | bias-reduced scores |

All three run under the judge rule's **`deterministic`** sampling profile — temp 0
+ a seed derived from the input — so the same target gets the **same ruling**
every time (reproducible, comparable over time).

---

## Pointwise

One model scores one target against the rubric.

```bash
curl -sX POST $CONDUCT/jobs -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{
  "task_type": "judge",
  "prompt": "(judge)",
  "inputs": { "mode": "pointwise", "target_job_id": "<uuid>", "apply_to_target": true }
}'
```

Verdict (on the judge job's `response` + `metadata.judge`):

```json
{ "mode": "pointwise", "score": 5, "rationale": "Relevant, correct, complete and clear.",
  "target_kind": "job", "applied_to_target": true }
```

With `apply_to_target`, the score is appended to the **target's**
`quality_scores` (`via="judge"`, `reviewer="<model>"`) — the same lane humans
use, so the [rollup](eval.md#reviewing-and-comparing) and the DPO export pick it
up with no extra work.

---

## Pairwise (order-swapped)

Compares two targets — e.g. a primary vs one of its shadows. The headline
feature is **position-bias defence**: each comparison runs **twice with A/B
swapped**, and a confident result requires *both* calls to name the same job as
winner. If they disagree, the judge is position-biased on that pair, so it
records a **tie** (no preference) rather than a coin-flip.

```bash
curl -sX POST $CONDUCT/jobs ... -d '{
  "task_type": "judge", "prompt": "(judge)",
  "inputs": { "mode": "pairwise", "target_job_id": "<A>", "against_job_id": "<B>", "apply_to_target": true }
}'
```

Verdict:

```json
{ "mode": "pairwise", "chosen_job_id": "<A>", "rejected_job_id": "<B>",
  "tie": false, "position_consistent": true,
  "rationale_ab": "...", "rationale_ba": "...", "applied_to_target": true }
```

> Pairwise stays **out of the 1-5 pointwise lane** (a relative win/loss isn't an
> absolute score). On `apply_to_target` it records the preference on *both*
> participants under a separate `pairwise_verdicts` metadata list, so it never
> skews per-model pointwise averages. That list is the (chosen, rejected) signal
> the DPO export reads.

---

## Panel / jury

A jury of **diverse models** each pointwise-scores the target; the result is the
**median** (robust to one odd juror). This is the most effective bias mitigation
in the literature, and Conduct gets two of its defences almost for free:

- **Self-preference exclusion.** Every job records `model_used`, so the jury
  drops any juror in the **same family** as the target's producer — *"never let
  gemma judge gemma."* Families are coarse vendor keys (gemma / llama / qwen /
  claude / mistral / …).
- **Sensitivity-aware jury.** A juror reads the target's content, so cloud
  jurors are dropped when the target is `confidential` (or `internal` without
  the client's cloud opt-in) — the routing engine's gate, applied per juror.

The jury is the judge rule's `preferred_model` + its `eval_shadow_models`
(repurposed as the panel, since judges don't fan out shadows), or an explicit
`inputs.panel`:

```bash
curl -sX POST $CONDUCT/jobs ... -d '{
  "task_type": "judge", "prompt": "(judge)",
  "inputs": { "mode": "panel", "target_job_id": "<uuid>", "apply_to_target": true,
              "panel": ["gemma4:e4b","llama3.2:3b","qwen3.5:4b"] }
}'
```

Verdict:

```json
{ "mode": "panel", "score": 3, "n": 2, "spread": 2, "disagreement": true,
  "panelists": [ {"model":"llama3.2:3b","score":4,"rationale":"…"},
                 {"model":"qwen3.5:4b","score":2,"rationale":"…"} ],
  "failures": [], "excluded": {"gemma4:e4b": "self-preference (same family as producer: gemma)"},
  "applied_to_target": true }
```

**Resilient:** a juror that errors or returns an unparseable verdict is recorded
in `failures` and skipped — the median is over the survivors. If *every* juror
fails, or the jury is empty after exclusions, the judge job fails loudly. A
`spread >= 2` sets `disagreement: true` — a contested verdict worth a human's
eyes. With `apply_to_target`, the median is written via `via="judge-panel"`.

---

## Where the scores go

| mode | write-back (on `apply_to_target`) | lane |
|------|-----------------------------------|------|
| pointwise | target.`quality_scores` | `via="judge"` |
| panel | target.`quality_scores` (the median) | `via="judge-panel"` |
| pairwise | both participants' `pairwise_verdicts` | win / loss records |

The `quality_scores` entries flow straight into the existing per-model rollup,
and all of it (scores + pairwise verdicts) is what the **dataset export**
turns into training data — removing the human-labelling bottleneck:

```bash
# DPO pairs from the pairwise judge's verdicts (or method=score for differentials)
curl -s "$CONDUCT/datasets/preferences?task_type=judge&method=pairwise" -H "Authorization: Bearer $ADMIN"
# SFT examples from high-scored responses (judge-sourced only)
curl -s "$CONDUCT/datasets/sft?min_score=4&via=judge&include_shadows=true" -H "Authorization: Bearer $ADMIN"
```

Both stream JSONL (`{prompt, system, chosen, rejected, meta}` /
`{prompt, system, completion, meta}`) ready for TRL/unsloth. Admin-only; see
`routes/datasets.py`.

## Safeguards (built in)

- **Determinism** — judge rules use the `deterministic` sampling profile, so
  rulings are reproducible. (See [the sampling profile](../config/seed.routing.yaml).)
- **Sensitivity inheritance** — a judge that resolved *looser* than its target
  fails **before any model call**, so a public judge can never send a
  confidential job's content to a cloud model. The panel applies this per juror.
- **Position-bias defence** — pairwise always order-swaps and requires agreement.
- **Self-preference defence** — the panel excludes the producer's family.
- **No recursion** — judges don't fan out eval shadows; don't give a judge rule
  cloud-shadow members you wouldn't want judged.
- **Robust parsing** — the verdict parser tolerates ```json fences + surrounding
  prose (a real failure mode on some local models) and rejects out-of-range or
  missing scores by failing the job, never recording a bogus number.

## Configuring the judge rule

A `judge` rule ships in [`config/seed.routing.yaml`](../config/seed.routing.yaml):

```yaml
- task_type: judge
  preferred_model: gemma4:e4b      # the single-model judge (pointwise/pairwise)
  fallback_model: gemma4:e4b
  sensitivity: internal            # floor — keeps internal/confidential content off cloud
  max_tokens: 400
  sampling: deterministic
  eval_shadow_models:              # repurposed as the PANEL jury
    - { model: llama3.2:3b, rate: 1.0 }
    - { model: qwen3.5:4b,  rate: 1.0 }
```

Tips:
- Edit it live with `PUT /routing/judge` (hot-reloads) or persist via `make seed`.
- For the panel, pick **diverse families** and ideally **`RESIDENT_MODELS`** —
  a non-resident Ollama juror triggers a model swap per run.
- The rubric is the judge task's prompt ([`prompts/shared/judge.md`](../prompts/shared/judge.md));
  override per request with `system_prompt`, and it's versioned like any prompt
  so scores stay comparable across rubric changes.

## Notes

- Judge jobs always run **async** (on the worker), never the sync API path.
- You can judge a **Job or a JobShadow** — shadows take their prompt + sensitivity
  from their parent.
- Build status: pointwise, pairwise, and panel all shipped (#17), and the
  `/datasets/sft` + `/datasets/preferences` JSONL export that consumes these
  scores is built (#16).
