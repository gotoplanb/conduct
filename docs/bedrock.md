# AWS Bedrock support

Conduct's `BedrockProvider` uses the Bedrock [Converse
API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)
— a unified request/response shape that AWS maps to each underlying
foundation model's native protocol. One provider class works for
**Anthropic-on-Bedrock**, **Llama**, **Mistral**, **Cohere**, **Nova**, and
**AI21** without per-family adapters.

## Auth model

**Per-client only, no host fallback.** Each Conduct client app stores its own
AWS credentials, encrypted at rest with `CONDUCT_SECRETS_KEY`. A client
without Bedrock creds simply can't route to Bedrock models — same strict
isolation rule as the existing per-client Anthropic API key. The host's
default credential chain (env vars, `~/.aws/`, instance profile) is **not**
consulted, because letting one client's job silently bill against the
operator's AWS account would defeat the per-tenant cost model.

Three values are stored together as a single Fernet-encrypted JSON blob:

- `access_key_id` — IAM access key id (e.g. `AKIAxxxxxxxxxxxxxxxx`)
- `secret_access_key` — paired secret
- `region` — the Bedrock region this key is enabled in (e.g. `us-west-2`)

Rotation is atomic at the blob level (any update overwrites all three).

## IAM policy

The minimum policy attached to the IAM user/role whose key Conduct holds:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockInvoke",
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:Converse"],
      "Resource": "*"
    }
  ]
}
```

Narrowing the `Resource` ARN to specific model IDs (e.g.
`arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-5-sonnet-*`)
is fine and recommended for production — Converse forwards the underlying
`InvokeModel` permission check, so blocking a model at the IAM layer
deterministically rejects routing to it.

**Cost control**: prefer setting an AWS Budgets alert on the IAM user (or a
service control policy on the account) rather than trying to cap spend
inside Conduct. Same delegation rationale as the per-client Anthropic key —
the cloud provider is the right place for cost ceilings.

## Model access

Bedrock requires you to explicitly request access to each foundation model
before its IDs are callable. In the AWS console:

1. Bedrock → **Model access** → **Modify model access**
2. Tick the boxes for the models you want — typically `anthropic.claude-*`
   plus whatever else you plan to A/B against.
3. Save. Anthropic models are typically approved instantly; some
   third-party models gate behind a use-case questionnaire.

You will get an `AccessDeniedException` on Converse calls until access is
granted in the same region as the credentials.

## Model IDs Conduct recognises

`provider_for_model` maps an ID to `bedrock` if either:

- it starts with one of the namespace prefixes:
  `anthropic.`, `amazon.`, `meta.`, `mistral.`, `cohere.`, `ai21.`,
  `deepseek.`, `stability.`
- or it starts with a cross-region inference profile prefix (`us.`, `eu.`,
  `apac.`, `us-gov.`) followed by one of the namespaces above
  (`us.anthropic.claude-3-5-sonnet-20241022-v2:0`).

Direct Anthropic API IDs (`claude-haiku-4-5`, `claude-sonnet-4-6`) still
route to the direct-API `AnthropicProvider`, **not** Bedrock. The dotted
namespace is how Conduct disambiguates "same model family, different
hosting" — and they have different per-client credential requirements.

Common production IDs (current as of 2026-06):

| Family | Direct API | Bedrock (flat) | Bedrock (cross-region) |
|---|---|---|---|
| Claude Haiku | `claude-haiku-4-5` | `anthropic.claude-3-5-haiku-20241022-v1:0` | `us.anthropic.claude-3-5-haiku-20241022-v1:0` |
| Claude Sonnet | `claude-sonnet-4-6` | `anthropic.claude-3-5-sonnet-20241022-v2:0` | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` |

> Verify with `aws bedrock list-foundation-models --region us-west-2` before
> wiring routing rules — Bedrock model IDs change frequently.

## Setup

### 1. Confirm `CONDUCT_SECRETS_KEY` is configured

Bedrock creds reuse the same master-key infrastructure as the Anthropic key.
If you haven't done this yet, see
[quickstart-ios-eval.md → Step 1](quickstart-ios-eval.md).

### 2. Set the creds on a client

**UI:** `/ui/clients` → click **Set Bedrock** on the client row → paste
access key id + secret access key + region → **Save Bedrock creds**.

**CLI:**

```bash
conduct clients set-bedrock-creds dave
# $EDITOR opens with a YAML template; fill in all three fields, save.
```

**API:**

```bash
curl -X PUT "$CONDUCT_BASE_URL/clients/$CLIENT_ID/bedrock-creds" \
  -H "Authorization: Bearer $CONDUCT_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "access_key_id": "AKIA...",
    "secret_access_key": "...",
    "region": "us-west-2"
  }'
```

### 3. Add pricing entries

Bedrock pricing is per-model per-region. Conduct reads it from
`config/pricing.yaml` under a `bedrock` top-level key. Cost lookup is keyed
on the exact Bedrock model id you used in routing, so the key under
`bedrock:` must match. Example:

```yaml
bedrock:
  anthropic.claude-3-5-sonnet-20241022-v2:0:
    input_per_1m_usd: 3.00
    output_per_1m_usd: 15.00
  us.anthropic.claude-3-5-sonnet-20241022-v2:0:
    input_per_1m_usd: 3.00
    output_per_1m_usd: 15.00
```

If an ID isn't in the file, the cost stays at `$0` (and Bedrock-side billing
still happens — only Conduct's recorded `cost_usd` is affected).

> Reload pricing without a restart by sending the API process `SIGHUP`. See
> [config/pricing.py](../config/pricing.py).

### 4. Wire a routing rule

Either edit `config/seed.routing.yaml` and re-seed, or update live:

```bash
conduct routing edit dad_joke
# add to eval_shadow_models:
#   - model: anthropic.claude-3-5-sonnet-20241022-v2:0
#     rate: 1.0
```

The same routing rule can reference both direct-API Anthropic and
Bedrock-Anthropic IDs simultaneously — Conduct treats them as independent
models for routing and A/B purposes. That makes Bedrock the natural place
to run regional or compliance-bound traffic side-by-side against your
existing direct-API setup.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `KeyError: client has no Bedrock credentials configured` | Client hasn't been set up yet, or the creds were cleared. Per-client only — no host fallback by design. |
| `AccessDeniedException` from Converse | Model access not granted in the AWS console for this region. Bedrock → Model access. |
| `ValidationException: model not found` | Model ID typo, or the ID exists only behind a cross-region inference profile (try the `us.` / `eu.` prefixed form). |
| `ThrottlingException` keeps happening | Bedrock has per-account per-model concurrent-request limits. Conduct retries up to 3× with exponential backoff; if it still throttles, request a quota increase in the Service Quotas console. |
| Cost shows `$0` for a Bedrock model in `/jobs/{id}/shadows` | Pricing entry missing under `bedrock:` in `config/pricing.yaml`. Billing on the AWS side is unaffected. |
| Cred set in UI but model still 503's the routing engine | UI flash succeeded but the API container is on a stale image. `make build && docker compose up -d api worker`. |
