# Quickstart: dad joke A/B from the Claude iOS app

End-to-end walkthrough: fire a dad-joke job from Claude on iOS, see every
model's attempt side-by-side, score the best one — without ever leaving the
chat.

It exercises every Conduct primitive in one flow:

- **Routing rule** with cross-family **eval shadows** (local + cloud)
- **Per-client Anthropic key** (encrypted at rest)
- **OAuth-protected MCP connector** Claude iOS talks to
- The five MCP tools: `list_task_types`, `create_job`, `get_job`,
  `list_shadows`, `submit_eval`

Total setup time: ~15 minutes if Conduct is already running locally.

> Prefer the CLI for everything? Every UI step here has a `conduct …`
> equivalent — both columns are shown.

## Prerequisites

1. **Conduct running locally.** `make up && make migrate && make seed`
   — see the [README](../README.md) and [deployment.md](deployment.md).
2. **A public HTTPS origin.** Claude iOS reaches Conduct over the public
   internet, so you need a tunnel or reverse proxy. The repo's docker-compose
   is local; pair it with ngrok / Cloudflare tunnel / etc. See
   [deployment.md](deployment.md) for the tunnel setup. Once it's live, set:
   - `CONDUCT_PUBLIC_URL=https://your.public.host` in `.env`
   - `UI_COOKIE_SECURE=true` in `.env`
   - `docker compose up -d api` to recreate the container

3. **An Anthropic API key.** Generate one at
   <https://console.anthropic.com/settings/keys>. Naming it after the client
   it'll belong to (e.g. `conduct_dave`) makes per-key budgets and audit easy.

## Step 1 — Master key for at-rest encryption

Conduct stores per-client Anthropic keys encrypted with a single master key
(Fernet, app-level). Generate one and add it to `.env`:

```bash
.venv/bin/python -c "from cryptography.fernet import Fernet; \
  print(f'CONDUCT_SECRETS_KEY={Fernet.generate_key().decode()}')" >> .env
docker compose up -d api worker      # recreate so the new env is loaded
```

> Back this key up — rotating it orphans every encrypted ciphertext in the
> DB. Clients would need their Anthropic keys re-pasted.

## Step 2 — A client app for the iOS connector

Each MCP connector binds to a Conduct client app. Jobs created over MCP are
attributed to that client and inherit its rate limits, cloud opt-in, and (now)
its Anthropic key.

**UI:** Go to `/ui/clients`, click **New client**, name it (e.g. `dave`),
check *Allow cloud models for internal-sensitivity jobs* if you'll want cloud
on non-public tasks too, **Create**. Copy the raw `cdt_…` API key from the
banner — it's shown once and never again.

**CLI:**

```bash
export CONDUCT_ADMIN_KEY=$(grep '^CONDUCT_ADMIN_KEY=' .env | cut -d= -f2-)
conduct clients create dave --allow-cloud-for-internal
```

## Step 3 — Set the client's Anthropic key

**UI:** On `/ui/clients`, click **Set Anthropic key** on the `dave` row, paste
the `sk-ant-…` key, **Save**. The plaintext never lands in the DB — only the
Fernet ciphertext and a timestamp.

**CLI:**

```bash
conduct clients set-anthropic-key dave    # opens $EDITOR; paste the key, save
```

Verify with `conduct clients list` — the row should show a recent
`anthropic_api_key_set_at` timestamp.

> If you skip this step, dad_joke still works — but the two cloud shadows
> (Haiku, Sonnet) won't fire for this client. Conduct's routing engine
> treats a client without an Anthropic key as "cloud unavailable" and falls
> back to local only.

> **AWS Bedrock variant.** Same flow with one swap: instead of (or in
> addition to) a direct-API Anthropic key, set per-client AWS creds and
> route to a Bedrock model id (`anthropic.claude-3-5-sonnet-20241022-v2:0`
> or the regional inference profile form like `us.anthropic.claude-...`).
> See [bedrock.md](bedrock.md) for the IAM policy, model-access flow, and
> pricing config. Routing supports mixing direct-API and Bedrock-hosted
> Anthropic in the same `eval_shadow_models` list, so you can A/B them
> head-to-head on the same dad joke.

## Step 4 — Confirm the dad_joke rule includes cloud shadows

`make seed` ships a `dad_joke` rule with a small-class local primary plus
peers. Make sure cloud shadows are in the eval-shadow list:

**CLI:**

```bash
conduct routing get dad_joke
```

You want `eval_shadow_models` to include both:

```yaml
- model: claude-haiku-4-5
  rate: 1.0
- model: claude-sonnet-4-6
  rate: 1.0
```

If they're missing, add them in `$EDITOR`:

```bash
conduct routing edit dad_joke
```

(Or PUT the JSON directly to `/routing/dad_joke` — see
[architecture.md](architecture.md).)

## Step 5 — Create the OAuth connector

**UI:** Go to `/ui/connectors`, click **New connector**, name it
`dave-ios`, bind it to the `dave` client app, **Create**. The page shows the
`client_id` + `client_secret` **once** — copy both into a password manager
immediately.

**CLI:**

```bash
conduct connectors create dave-ios --client dave
```

## Step 6 — Add the connector in Claude iOS

1. In the Claude app, open **Settings → Connectors → Add custom connector**.
2. Enter your Conduct's `/mcp` endpoint:
   `https://your.public.host/mcp`
3. Paste the `client_id` + `client_secret` from Step 5.
4. Tap **Connect**. You'll be bounced to Conduct's `/oauth/authorize` page —
   approve by logging in with `CONDUCT_ADMIN_KEY`.
5. After redirect-back, Claude lists the five tools:
   `list_task_types`, `list_jobs`, `get_job`, `create_job`, `list_shadows`,
   `submit_eval`. If `list_shadows` isn't there, the connector cached an older
   tool list — disconnect and reconnect once.

## Step 7 — The conversation

Open a new chat in Claude iOS with the `dave-ios` connector enabled. Try these
prompts in order — Claude will pick the right tool on its own:

1. **Discover the menu.**
   > "What task types does my Conduct instance know about?"

   Claude calls `list_task_types`. You should see `dad_joke` with
   `preferred_model: llama3.2:3b`, sensitivity `public`.

2. **Fire the job.**
   > "Create a dad joke job about databases and force all shadows."

   Claude calls `create_job(task_type="dad_joke", prompt="…",
   force_shadows=True)`. `force_shadows=True` makes every eligible candidate
   fire for *this* request even if its sampling rate is below 1.0 — useful
   when you specifically want a full A/B.

   For dad_joke the primary (`llama3.2:3b`) is small and resident-local, so
   Conduct runs it inline and Claude gets the punchline back in the first
   response.

3. **Pull the shadows.**
   > "Show me every shadow's response for that job."

   Claude calls `list_shadows(job_id="…")`. You get the side-by-side: three
   local models, plus Haiku 4.5 and Sonnet 4.6 each with their own response,
   latency, and cost. Costs for the cloud shadows are billed against
   `dave`'s per-client Anthropic key — visible in your Anthropic console.

4. **Score the parent.**
   > "That was a 4 — Sonnet's was the funniest. Submit the eval."

   Claude calls `submit_eval(job_id="…", score=4, note="Sonnet's was the
   funniest")`. The score lands on the parent Job and rolls up into the
   per-task quality stats at `/ui/tasks/dad_joke`.

5. **Score the individual shadows** *(optional but where the real eval value
   lives).*
   > "Give Sonnet a 5, Haiku a 4, and both gemma4 attempts a 2 — Sonnet's
   > punchline was the only one I actually laughed at."

   Claude calls `submit_eval` once per shadow, passing each shadow's
   `shadow_id` (from step 3) as the `job_id` argument. The server figures
   out that those UUIDs are shadows and stores the scores on the shadow
   rows, scoped to your client. Per-model averages then show up at
   `/ui/tasks/dad_joke` for the model-comparison view.

That's it — full A/B + scoring loop from the phone, with cloud costs
attributed to a single per-client Anthropic key you can budget independently
of any other tenant.

## What's happening under the hood

- The **routing engine** ([routing/engine.py](../routing/engine.py)) picks the
  primary per the rule. `cloud_available=True` because `dave` has an
  Anthropic key set — without it, cloud candidates are silently skipped.
- The **provider registry** ([providers/registry.py](../providers/registry.py))
  resolves Anthropic per client: `get_for_client(dave, "anthropic")` builds a
  fresh `AnthropicProvider` with `dave`'s decrypted key for the call, then
  drops it.
- **Shadows fan out in parallel** after the primary completes — see
  [eval/fanout.py](../eval/fanout.py) and [eval/shadow_runner.py](../eval/shadow_runner.py).
- Every span shows up in **Grafana Tempo** with `job.client_app=dave` and
  `model.used=claude-sonnet-4-6` (etc.) — useful for "what did sonnet cost
  this week" queries. See [observability.md](observability.md).

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI says "CONDUCT_SECRETS_KEY is not configured" | Step 1 wasn't run, or the container wasn't recreated after editing `.env`. `docker compose up -d api worker`. |
| Cloud shadows show `$0` cost | The model isn't in `config/pricing.yaml`. Add an entry and rebuild the image (templates and config are baked at build time). |
| iOS connector doesn't see `list_shadows` | Tool list was cached. Disconnect and reconnect the connector. |
| `decide()` raises `SensitivityViolation` for a dad_joke | The rule's sensitivity floor got raised above `public`. `conduct routing get dad_joke` to inspect, `edit` to fix. |
| Job is `complete` but no shadows show up | The rule's `eval_shadow_models` is empty, or every shadow's `rate` is 0. Use `force_shadows=true` to override sampling. |
