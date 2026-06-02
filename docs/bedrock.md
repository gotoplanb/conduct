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

Two auth styles are supported. The encrypted blob holds one of:

**(A) Long-term Bedrock API key (recommended).** Generated in the Bedrock
console under *API keys → Long-term API keys*; no IAM user, no AKID/secret
pair to manage. The blob is `{bearer_token, region}` — two fields,
generated and revoked from one place.

**(B) IAM access key + secret.** Traditional AWS SigV4 auth. The blob is
`{access_key_id, secret_access_key, region}` — three fields. Useful if you
already have IAM-based provisioning automation and want to keep Bedrock on
the same key as other AWS services.

In code: bearer tokens are injected via a botocore event hook
(`before-send.bedrock-runtime.*`) that overwrites the Authorization header
per request. **Conduct deliberately does not use the documented
`AWS_BEARER_TOKEN_BEDROCK` env var**, because env vars are process-global
and would race under concurrent multi-tenant shadow fan-out.

Rotation is atomic at the blob level (any PUT overwrites the whole thing,
so you can switch a client from access-key to bearer-token by re-saving).

## IAM policy

For both auth styles, the underlying identity (whether the IAM user behind
an access-key pair or the IAM identity behind a long-term API key) needs:

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

For long-term API keys: the Bedrock console binds each key to a single IAM
identity at creation time. Manage permissions on that identity exactly as
above; revoking the key in the console invalidates it immediately.

**Cost control**: prefer setting an AWS Budgets alert on the IAM identity
(or a service control policy on the account) rather than trying to cap
spend inside Conduct. Same delegation rationale as the per-client
Anthropic key — the cloud provider is the right place for cost ceilings.

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
- or it starts with a cross-region inference profile prefix
  (`us.`, `eu.`, `au.`, `jp.`, `apac.`, `us-gov.`, `global.`) followed by
  one of the namespaces above (`us.anthropic.claude-sonnet-4-6`,
  `global.anthropic.claude-haiku-4-5-20251001-v1:0`).

Direct Anthropic API IDs (`claude-haiku-4-5`, `claude-sonnet-4-6`) still
route to the direct-API `AnthropicProvider`, **not** Bedrock. The dotted
namespace is how Conduct disambiguates "same model family, different
hosting" — and they have different per-client credential requirements.

### Claude 4.x on Bedrock (verified 2026-06-02)

| Family | Direct-API ID | Bedrock flat ID | Bedrock geo IDs | Bedrock global ID |
|---|---|---|---|---|
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | `anthropic.claude-sonnet-4-6` | `us.anthropic.claude-sonnet-4-6`<br>`eu.anthropic.claude-sonnet-4-6`<br>`au.anthropic.claude-sonnet-4-6`<br>`jp.anthropic.claude-sonnet-4-6` | `global.anthropic.claude-sonnet-4-6` |
| Claude Haiku 4.5 | `claude-haiku-4-5` | `anthropic.claude-haiku-4-5-20251001-v1:0` | `us.anthropic.claude-haiku-4-5-20251001-v1:0`<br>`eu.anthropic.claude-haiku-4-5-20251001-v1:0`<br>`au.anthropic.claude-haiku-4-5-20251001-v1:0`<br>`jp.anthropic.claude-haiku-4-5-20251001-v1:0` | `global.anthropic.claude-haiku-4-5-20251001-v1:0` |

Two non-obvious wrinkles AWS hits you with:

1. **Sonnet 4.6 uses naked IDs; Haiku 4.5 is date-stamped.** This is just
   how AWS chose to namespace them — there's no pattern to predict, so
   verify before pasting into a routing rule.
2. **In-region availability is narrower than you'd expect.** Sonnet 4.6 is
   *only* directly callable in `eu-west-2 (London)`; everywhere else you
   need a geo or global inference profile id. Haiku 4.5 is directly
   callable in `us-east-1`, `eu-north-1`, `eu-west-1`, `ap-northeast-1`,
   `ap-southeast-4`. The geo IDs (`us.…`, `eu.…`) work from a much wider
   set of regions — see the AWS docs model card for the source-region
   matrix.

**Recommendation for US-region AWS accounts:** use the geo IDs
(`us.anthropic.claude-sonnet-4-6` and
`us.anthropic.claude-haiku-4-5-20251001-v1:0`). They work from any of
`us-east-1`, `us-east-2`, `us-west-1`, `us-west-2`, `ca-central-1`,
`ca-west-1`, and let Bedrock distribute load across the underlying
regions — better latency variance and quota headroom than pinning to a
single in-region flat ID.

> Source of truth: the model cards under
> `https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-*.html`.
> Verify with `aws bedrock list-foundation-models --region us-east-1`
> before wiring routing rules — IDs change as new versions ship.

## Setup

### 1. Confirm `CONDUCT_SECRETS_KEY` is configured

Bedrock creds reuse the same master-key infrastructure as the Anthropic key.
If you haven't done this yet, see
[quickstart-ios-eval.md → Step 1](quickstart-ios-eval.md).

### 2. Set the creds on a client

#### Long-term API key (recommended)

Generate one in the Bedrock console → *API keys* → *Long-term API keys*. No
AWS CLI or IAM provisioning needed; the key is bound to the IAM identity
you select at creation time.

**UI:** `/ui/clients` → **Set Bedrock** → leave the *Long-term API key*
radio selected → paste the key + region → **Save Bedrock creds**.

**CLI:**

```bash
conduct clients set-bedrock-creds dave
# $EDITOR opens with a YAML template — fill in `bearer_token` and
# `region`, leave the access-key fields blank, save.
```

**API:**

```bash
curl -X PUT "$CONDUCT_BASE_URL/clients/$CLIENT_ID/bedrock-creds" \
  -H "Authorization: Bearer $CONDUCT_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"bearer_token": "ABSK...", "region": "us-east-1"}'
```

#### Access key + secret (traditional)

Create an IAM user with the policy above, generate a programmatic access
key pair, then:

**UI:** `/ui/clients` → **Set Bedrock** → switch to the *Access key +
secret* radio → paste both values + region → **Save Bedrock creds**.

**CLI:** same `conduct clients set-bedrock-creds dave`, fill in
`access_key_id` + `secret_access_key` + `region` instead, leave
`bearer_token` blank.

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
| Long-term API key returns `401 Unauthorized` | Key was revoked in the Bedrock console, or it was generated for a different region than what's stored on the client. Re-issue and re-save. |
| `AccessDeniedException` from Converse | Model access not granted in the AWS console for this region. Bedrock → Model access. |
| `ValidationException: model not found` | Model ID typo, or the ID exists only behind a cross-region inference profile (try the `us.` / `eu.` prefixed form). |
| `ThrottlingException` keeps happening | Bedrock has per-account per-model concurrent-request limits. Conduct retries up to 3× with exponential backoff; if it still throttles, request a quota increase in the Service Quotas console. |
| Cost shows `$0` for a Bedrock model in `/jobs/{id}/shadows` | Pricing entry missing under `bedrock:` in `config/pricing.yaml`. Billing on the AWS side is unaffected. |
| Cred set in UI but model still 503's the routing engine | UI flash succeeded but the API container is on a stale image. `make build && docker compose up -d api worker`. |
