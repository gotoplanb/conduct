# Conduct — LLM Dispatch Service

## What It Is

Conduct is a multi-tenant LLM dispatch service that routes AI workloads to the right model — local or cloud — based on task type, sensitivity, and model availability. It provides a unified API surface for any client application to submit jobs without knowing which model will handle them.

Built with FastAPI and Postgres. Runs on the M5 MacBook Pro. Accessible via ngrok HTTPS tunnel.

---

## Design Principles

- **Invisible to end users** — client apps call Conduct, humans see results
- **Multi-tenant from day one** — every job is attributed to a client app
- **Eval data as a side effect** — every job execution generates routing intelligence automatically
- **Sync when possible, async when necessary** — fast path for loaded models, queue for everything else
- **Local first, cloud fallback** — prefer local inference, escalate to cloud when task warrants it

---

## Tech Stack

- **FastAPI** — API gateway and route handlers
- **Postgres** — job records, client apps, usage tracking, routing config
- **Redis + RQ** — async job queue and worker
- **Ollama** — local model management and inference
- **Anthropic SDK** — Claude cloud models (Haiku, Sonnet, Opus)
- **AWS Boto3** — Bedrock models for cloud-provider-diverse workloads
- **ngrok** — HTTPS tunnel for external access

---

## Data Models

### ClientApp

```python
class ClientApp(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    name = models.CharField(max_length=100)       # arbitrary identifier per deployment
    api_key_hash = models.CharField(max_length=64, unique=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True)
```

API keys are stored hashed (SHA-256). Raw key shown once on creation, never again.

### Job

```python
class Job(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('complete', 'Complete'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    SENSITIVITY_CHOICES = [
        ('standard', 'Standard'),    # cloud models acceptable
        ('sensitive', 'Sensitive'),  # local model required
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    client_app = models.ForeignKey(ClientApp, on_delete=models.CASCADE)
    task_type = models.CharField(max_length=100)   # 'bio_generation', 'email_classification', etc.
    sensitivity = models.CharField(max_length=20, choices=SENSITIVITY_CHOICES, default='standard')
    priority = models.IntegerField(default=5)       # 1 = highest, 10 = lowest
    prompt = models.TextField()
    system_prompt = models.TextField(blank=True)
    model_requested = models.CharField(max_length=100, blank=True)  # optional override
    model_used = models.CharField(max_length=100, blank=True)
    response = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    tokens_in = models.IntegerField(null=True, blank=True)
    tokens_out = models.IntegerField(null=True, blank=True)
    cost_usd = models.DecimalField(max_digits=10, decimal_places=6, null=True, blank=True)
    latency_ms = models.IntegerField(null=True, blank=True)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)  # arbitrary client context
```

### RoutingRule

```python
class RoutingRule(models.Model):
    task_type = models.CharField(max_length=100, unique=True)
    preferred_model = models.CharField(max_length=100)   # 'llama3.3:70b', 'claude-sonnet-4-5', etc.
    fallback_model = models.CharField(max_length=100)
    sensitivity = models.CharField(max_length=20, default='standard')
    max_tokens = models.IntegerField(default=1000)
    notes = models.TextField(blank=True)
    updated_at = models.DateTimeField(auto_now=True)
```

### ClientAppUsage

```python
class ClientAppUsage(models.Model):
    client_app = models.ForeignKey(ClientApp, on_delete=models.CASCADE)
    date = models.DateField()
    tokens_in = models.BigIntegerField(default=0)
    tokens_out = models.BigIntegerField(default=0)
    job_count = models.IntegerField(default=0)
    cost_usd = models.DecimalField(max_digits=10, decimal_places=4, default=0)

    class Meta:
        unique_together = ('client_app', 'date')
```

---


---

## Data Sensitivity Framework

Rather than mapping to specific compliance regimes (PCI, HIPAA) which may not apply to most customers, Conduct uses a simple three-tier sensitivity model:

| Level | Definition | Model Routing |
|-------|------------|---------------|
| `public` | Data intended for publication. Agent bios, marketing copy, public listings. | Cloud models acceptable |
| `internal` | Business operational data. Agent PII, email content, task data. | Local preferred, cloud acceptable with client consent |
| `confidential` | Financial records, commission calculations, anything approaching regulated data. | Local only, never cloud |

Clients set sensitivity per job submission. Routing rules enforce the constraint — a `confidential` job will never be routed to a cloud provider regardless of model availability or fallback logic.

**Bio generation** is `internal` — even when the output goes on a public website, processing PII locally is the cleaner default and costs nothing extra on local hardware.

**Email classification** is `internal` — email content is business-sensitive but not financial.

**Commission audit** is `confidential` — financial data stays local, period.

If a customer operates under a specific compliance framework (HIPAA, SOC2, etc.) and needs mapping from their framework to Conduct's sensitivity tiers, that's a consulting conversation — not something Conduct enforces automatically.

## API Routes

All routes require `Authorization: Bearer {api_key}` header.

### Job Submission

```
POST /jobs
```

Request body:
```json
{
  "task_type": "bio_generation",
  "prompt": "Generate a bio for...",
  "system_prompt": "You are...",
  "sensitivity": "standard",
  "priority": 5,
  "model": null,
  "metadata": {}
}
```

**Sync path** (model loaded, queue empty or priority 1):
- Execute immediately
- Return `200` with full response

```json
{
  "job_id": "uuid",
  "status": "complete",
  "response": "...",
  "model_used": "llama3.3:70b",
  "tokens_in": 245,
  "tokens_out": 312,
  "latency_ms": 4200,
  "cost_usd": 0.000000
}
```

**Async path** (model not loaded, queue has depth, or async requested):
- Create job record
- Enqueue in RQ
- Return `202` with job ID

```json
{
  "job_id": "uuid",
  "status": "pending",
  "poll_url": "/jobs/uuid"
}
```

---

```
GET /jobs/{job_id}
```

Returns current job status and result if complete.

```
DELETE /jobs/{job_id}
```

Cancel a pending job.

---

### Model Management

```
GET /models
```

Returns all installed Ollama models with loaded status and memory usage, plus configured cloud models.

```json
{
  "local": [
    {
      "name": "llama3.3:70b",
      "status": "loaded",
      "size_gb": 40.2,
      "last_used": "2026-04-29T12:00:00Z"
    },
    {
      "name": "qwen2.5:72b",
      "status": "unloaded",
      "size_gb": 43.1,
      "last_used": null
    }
  ],
  "cloud": [
    {"name": "claude-haiku-4-5", "provider": "anthropic"},
    {"name": "claude-sonnet-4-5", "provider": "anthropic"},
    {"name": "claude-opus-4-5", "provider": "anthropic"}
  ]
}
```

```
POST /models/{model_name}/load
```

Explicitly load a local model into memory. Returns 202 if load takes time.

```
POST /models/{model_name}/unload
```

Free a model from memory.

---

### Routing Config

```
GET /routing
```

Returns current task_type → model mapping.

```
PUT /routing/{task_type}
```

Update routing for a specific task type. Hot reload — no restart required.

```json
{
  "preferred_model": "qwen2.5:72b",
  "fallback_model": "claude-sonnet-4-5",
  "sensitivity": "standard",
  "max_tokens": 1000
}
```

---

### Observability

```
GET /health
```

Service health, loaded models, queue depth, worker status.

```
GET /metrics
```

Aggregated stats. Optional query params: `?client_app_id=uuid&days=30&task_type=bio_generation`

```json
{
  "period_days": 30,
  "total_jobs": 847,
  "jobs_by_status": {"complete": 831, "failed": 12, "pending": 4},
  "jobs_by_model": {
    "llama3.3:70b": {"count": 612, "avg_latency_ms": 3800, "total_cost_usd": 0.00},
    "claude-sonnet-4-5": {"count": 219, "avg_latency_ms": 1200, "total_cost_usd": 4.32}
  },
  "jobs_by_task_type": {
    "bio_generation": {"count": 203, "preferred_model": "llama3.3:70b"},
    "email_classification": {"count": 441, "preferred_model": "llama3.3:70b"}
  }
}
```

---

### Client App Management

```
POST /clients               — create new client app, returns raw API key once
GET  /clients               — list all client apps
GET  /clients/{id}/usage    — usage summary for a specific client
PATCH /clients/{id}         — update name or active status
```

Client management routes require an admin API key (separate from client keys, set via environment variable).

---

## Routing Logic

On job submission, Conduct selects a model using this priority order:

1. **Explicit model override** — if `model` field provided in request, use it (if available)
2. **Sensitivity check** — if `sensitive`, only local models eligible
3. **Routing rule lookup** — find `RoutingRule` for `task_type`
4. **Model availability** — is preferred model loaded? If yes, use it
5. **Swap decision** — if preferred model not loaded:
   - If queue is empty and job is high priority → swap model, execute (log swap latency)
   - Otherwise → enqueue for worker to handle when model is available or swapped
6. **Fallback** — if preferred model unavailable and sensitivity allows, use fallback model
7. **Default** — if no routing rule exists, use configured default model

---

## Worker

Single RQ worker process. Responsibilities:

- Pull jobs from queue in priority order
- Check which model is currently loaded
- Swap model if needed (via Ollama API)
- Execute inference
- Write result back to Job record
- Update ClientAppUsage daily aggregate
- Log swap time as part of job metadata

Worker runs as a persistent background process, managed by a launchd plist on macOS.

---

## Eval Framework

Conduct generates eval data as a side effect of normal operation. No separate benchmark runs needed.

```
GET /eval/compare
```

Query params: `?task_type=bio_generation&days=30`

Returns side-by-side comparison of all models used for a given task type:

```json
{
  "task_type": "bio_generation",
  "models": [
    {
      "model": "llama3.3:70b",
      "job_count": 145,
      "avg_latency_ms": 4100,
      "avg_tokens_out": 287,
      "cost_per_job_usd": 0.000,
      "failure_rate": 0.007
    },
    {
      "model": "claude-sonnet-4-5",
      "job_count": 58,
      "avg_latency_ms": 980,
      "avg_tokens_out": 312,
      "cost_per_job_usd": 0.0031,
      "failure_rate": 0.000
    }
  ]
}
```

Manual quality scoring can be added later — a `POST /jobs/{id}/score` endpoint that accepts a 1-5 quality rating, enabling human-in-the-loop eval without changing the core job flow.

---

## Authentication

- Client API keys: 32-byte random, stored SHA-256 hashed, shown once on creation
- Admin key: set via `CONDUCT_ADMIN_KEY` environment variable
- All requests validated via FastAPI dependency before hitting route logic
- Rate limiting per client app configurable in RoutingRule or ClientApp record

---

## Configuration

All via environment variables:

```bash
DATABASE_URL=postgresql://...
REDIS_URL=redis://localhost:6379
OLLAMA_BASE_URL=http://localhost:11434
ANTHROPIC_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
CONDUCT_ADMIN_KEY=...
NGROK_AUTHTOKEN=...
DEFAULT_MODEL=llama3.3:70b
DEFAULT_SENSITIVE_MODEL=llama3.3:70b
```

---

## Initial Routing Table (seed data)

| task_type | preferred_model | fallback_model | sensitivity |
|-----------|----------------|----------------|-------------|
| bio_generation | llama3.3:70b | claude-sonnet-4-5 | standard |
| email_classification | llama3.3:70b | claude-haiku-4-5 | standard |
| email_extraction | qwen2.5:72b | claude-sonnet-4-5 | standard |
| agenda_generation | llama3.3:70b | claude-sonnet-4-5 | standard |
| commission_audit | llama3.3:70b | llama3.3:70b | sensitive |
| ledger_calculation | llama3.3:70b | llama3.3:70b | sensitive |
| code_generation | claude-sonnet-4-5 | claude-opus-4-5 | standard |
| sre_triage | llama3.3:70b | claude-sonnet-4-5 | standard |

---


---

## Observability

Instrumented from day one. No bolting on later.

### OpenTelemetry

Use the OpenTelemetry Python SDK for traces and metrics. Every job execution gets a trace. Every provider call gets a child span. Routing decisions are logged as span events.

```python
# Every job gets a root span
with tracer.start_as_current_span("conduct.job") as span:
    span.set_attribute("job.id", str(job.id))
    span.set_attribute("job.task_type", job.task_type)
    span.set_attribute("job.sensitivity", job.sensitivity)
    span.set_attribute("job.client_app", job.client_app.name)
    span.set_attribute("model.requested", model_requested)
    span.set_attribute("model.used", model_used)
    span.set_attribute("model.provider", provider)  # local/anthropic/bedrock

    # Child span for the actual inference call
    with tracer.start_as_current_span("conduct.inference"):
        response = await provider.complete(...)
        span.set_attribute("tokens.in", tokens_in)
        span.set_attribute("tokens.out", tokens_out)
        span.set_attribute("cost.usd", cost_usd)
        span.set_attribute("latency.ms", latency_ms)
```

### Prometheus Metrics

Expose `GET /metrics/prometheus` for Grafana Alloy to scrape. Separate from the JSON `/metrics` endpoint.

```
conduct_jobs_total{status, task_type, model, client_app}
conduct_job_duration_seconds{task_type, model}
conduct_tokens_total{direction, model, client_app}
conduct_model_load_duration_seconds{model}
conduct_model_swap_total{from_model, to_model}
conduct_queue_depth{priority}
conduct_cost_usd_total{model, client_app}
conduct_retry_total{provider, reason}
conduct_fallback_total{from_provider, to_provider, reason}
```

### Grafana Alloy

Configure Alloy to scrape `http://localhost:8000/metrics/prometheus` and forward to the local LGTM stack (Loki, Mimir, Tempo). Same pattern as Watchtower — use existing Alloy config as the template.

The Grafana dashboard surfaces: job throughput by model, cost by client app, latency distribution, queue depth over time, model swap frequency, fallback rate by provider. This is also the live eval dashboard.

---

## Provider SDK Strategy

Use native SDKs directly — not an abstraction library like `llm`. Reasons:

- Precise token counts and cost attribution require direct access to raw response objects
- Each provider has different retry semantics — abstracting them loses that control
- Conduct's routing layer IS the abstraction — no need to wrap a wrapper
- Direct SDKs mean one less dependency between the job executor and the response

Use Simon Willison's `llm` CLI as an exploration tool for testing new models and prompts before wiring them into Conduct. Not as a runtime dependency.

### Provider Implementations

Each provider lives in `providers/` and implements a common interface:

```python
class BaseProvider:
    async def complete(self, prompt: str, system_prompt: str, 
                      max_tokens: int, **kwargs) -> ProviderResponse:
        ...

class ProviderResponse:
    response: str
    tokens_in: int
    tokens_out: int
    cost_usd: Decimal
    latency_ms: int
    model_used: str
```

---

## Retry and Fallback Strategy

Retry logic is model-aware, not generic HTTP retry. Each failure mode has a different response.

### Static Retry Layer (v1)

Use `tenacity` for provider-specific retry policies:

```python
# Anthropic — exponential backoff on rate limits
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type(RateLimitError)
)
async def call_anthropic(...): ...

# Ollama — short timeout, no retry (fall through to fallback logic)
@retry(
    stop=stop_after_attempt(1),
    wait=wait_fixed(0)
)
async def call_ollama(...): ...

# Bedrock — longer timeout for cold starts
@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(5),
    retry=retry_if_exception_type(TimeoutError)
)
async def call_bedrock(...): ...
```

Failure modes and responses:

| Failure | Response |
|---------|----------|
| Ollama timeout | Fall back to cloud model if sensitivity allows |
| Ollama model not loaded | Enqueue for worker to swap and retry |
| Anthropic rate limit | Exponential backoff, retry same provider |
| Anthropic unavailable | Fall back to local model |
| Bedrock cold start | Retry with longer timeout |
| All providers failed | Return 503 with retry-after header |

### Intelligent Triage Layer (v2)

A small always-resident local model (3B-8B, ~6GB memory footprint) watches failure signals and makes routing decisions. Same pattern as Hermit Watch SRE triage — cheap, fast, always local, never sees prompt content, only operational metadata.

```python
triage_context = {
    "task_type": job.task_type,
    "error_type": error.__class__.__name__,
    "error_message": str(error),
    "attempted_model": model_used,
    "available_models": get_loaded_models(),
    "queue_depth": get_queue_depth(),
    "client_priority": job.priority,
    "sensitivity": job.sensitivity
}

decision = await triage_model.complete(
    prompt=json.dumps(triage_context),
    system_prompt=TRIAGE_SYSTEM_PROMPT  # loaded from prompts/ in GitHub
)
# Returns: retry_local / swap_model / fallback_cloud / return_503 / escalate
```

Triage decisions feed back into the eval framework — did the job succeed after the triage model's recommendation? That's free reinforcement signal.

Design the failure handling interface in v1 so the triage model plugs in without a refactor. The static retry logic and the intelligent triage layer share the same interface — swap one for the other when ready.

The triage model runs on essentially free compute since the hardware is already paid for. LLM all the things.

---

## Prompt Library

Prompts live in a `prompts/` directory in the GitHub repo, versioned as markdown files. Clients submit a `task_type` only — they never see, know about, or send prompts. Conduct resolves the correct prompt internally based on `task_type` and `client_app_id`.

This means:
- Improving a prompt is a Conduct-side change with zero client code changes
- Every job log references the exact prompt file and git commit hash — full lineage
- Prompt IP stays inside Conduct, not scattered across client codebases
- Observability is clean — any output can be traced back to the exact prompt version that produced it

### Prompt Resolution Hierarchy

Conduct resolves prompts using client-specific override with shared fallback:

1. Check `prompts/clients/{client_name}/{task_type}.md` — client-specific override
2. Fall back to `prompts/shared/{task_type}.md` — shared default

The `client_name` is derived from the authenticated `ClientApp` record via bearer token. Client never knows which path was resolved.

### Folder Structure

```
prompts/
├── shared/                          — available to all clients
│   ├── email_classification.md
│   ├── email_extraction.md
│   ├── bio_generation.md
│   ├── agenda_generation.md
│   ├── commission_audit.md
│   ├── sre_triage.md
│   ├── triage.md                    — Conduct's own v2 failure triage prompt
│   └── README.md                    — prompt authoring guidelines
└── clients/                        — gitignored; deployment-specific overrides
    └── {client_name}/
        └── {task_type}.md           — overrides the matching shared prompt for that client
```

### Multi-Tenancy and Prompt Isolation

Prompt isolation concerns resolve by customer segment:

- **Small/medium customers** — shared Conduct instance, client folder overrides provide customization, prompts coexist safely since no client can read another's folder
- **Customers with proprietary prompt IP or compliance requirements** — they run Conduct themselves via the open source repo, use the managed service only for periodic evals

Nobody exists in the middle. If you care enough about prompt isolation to be uncomfortable with shared hosting, you're technical enough to self-host. The open source option removes the objection entirely.

### Hot Reload

Conduct reads prompts from the filesystem at request time — no restart required when prompts change. A prompt improvement is a git pull on the Conduct host. All subsequent jobs use the improved version automatically.

Every job execution logs:
- Prompt file path resolved
- Git commit hash of the prompt file at execution time
- Whether shared or client-specific prompt was used

Full audit trail. Any output traceable to the exact prompt version that produced it.

---

## Project Structure

```
conduct/
├── main.py                  — FastAPI app, route registration
├── auth.py                  — API key validation dependency
├── models/
│   ├── job.py
│   ├── client.py
│   ├── routing.py
│   └── usage.py
├── routes/
│   ├── jobs.py
│   ├── models.py
│   ├── routing.py
│   ├── clients.py
│   ├── metrics.py           — JSON metrics + Prometheus endpoint
│   └── health.py
├── worker/
│   ├── queue.py             — RQ setup
│   ├── executor.py          — model execution logic
│   └── model_manager.py     — Ollama model load/unload
├── providers/
│   ├── base.py              — BaseProvider interface
│   ├── ollama.py            — local inference
│   ├── anthropic.py         — Claude cloud
│   └── bedrock.py           — AWS Bedrock
├── retry/
│   ├── static.py            — tenacity-based provider retry policies
│   └── triage.py            — v2 intelligent triage model interface
├── eval/
│   └── compare.py           — eval aggregation queries
├── observability/
│   ├── tracing.py           — OpenTelemetry tracer setup
│   └── metrics.py           — Prometheus metrics definitions
├── prompts/
│   ├── shared/
│   │   ├── bio_generation.md
│   │   ├── email_classification.md
│   │   ├── email_extraction.md
│   │   ├── agenda_generation.md
│   │   ├── commission_audit.md
│   │   ├── sre_triage.md
│   │   ├── triage.md
│   │   └── README.md
│   └── clients/             — gitignored; deployment-specific overrides
├── alembic/                 — database migrations
├── tests/
├── spec.md                  — this file
└── CLAUDE.md                — session context for Claude Code
```

---

## Testing

- Auth: invalid key returns 403, inactive client returns 403
- Sync path: loaded model returns 200 with response
- Async path: unloaded model returns 202 with job_id
- Routing: sensitive job never routes to cloud model
- Worker: job transitions through pending → running → complete
- Usage: ClientAppUsage updates after job completion
- Eval: compare endpoint returns correct aggregations

---

## Definition of Done

- `POST /jobs` accepts requests and routes correctly (sync and async)
- Worker processes queued jobs and writes results
- All model management routes operational
- Routing config hot-reloadable via API
- `/metrics` and `/eval/compare` return meaningful data after 10+ jobs
- A registered client app can submit a bio generation job and get a response
- ngrok HTTPS tunnel accessible from external services
- At least one client app registered via `make seed`
- Tests pass

---

## Out of Scope

- Multi-machine worker distribution (single machine for now)
- Streaming responses (batch only for v1)
- Fine-tuning or model training
- Billing/invoicing (usage tracking is the foundation, billing layer is future)
