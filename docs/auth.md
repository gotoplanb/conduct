# Auth, clients, and sensitivity tiers

Conduct has two kinds of caller — **client apps** (submit jobs) and the **admin**
(manage the instance) — plus a UI session and OAuth tokens that both resolve
back to those. Data egress is governed by **sensitivity tiers** enforced in the
routing engine.

## Principals

| Principal | Credential | Used for |
|---|---|---|
| **Client app** | `Authorization: Bearer cdt_<key>` | `POST /jobs`, `GET/DELETE /jobs/{id}`, `POST /tts` |
| **Admin** | `Authorization: Bearer <CONDUCT_ADMIN_KEY>` | client / routing / prompt / eval / model management, JSON metrics |
| **UI session** | `conduct_admin` cookie (the admin key) | `/ui/*` pages, OAuth consent |
| **MCP connector** | OAuth bearer (`cdt_at_<token>`) | `/mcp` — resolves to a bound client app |

Open (no auth): `GET /health`, `GET /metrics/prometheus`, the OAuth discovery
endpoints. See [mcp-connector.md](mcp-connector.md) for the OAuth flow.

## Client API keys

- **Format:** `cdt_<random>`. Only a SHA-256 **hash** is stored
  (`client_apps.api_key_hash`) — the raw key is shown **once** at creation and is
  unrecoverable.
- **Minted by:** `make seed` (from `config/seed.clients.yaml`), `POST /clients`,
  or the **Clients** UI.
- **Verification:** the bearer token is hashed and looked up; an inactive client
  (`is_active=false`) is rejected. Auth failures return `403`.
- **Rotation / revocation:** rotate the key or toggle `is_active` from the
  Clients UI or `PATCH /clients/{id}`. Rotating invalidates the old key
  immediately; deactivating blocks the client (and, for MCP, its OAuth tokens).

### Per-client attributes

| Field | Effect |
|---|---|
| `is_active` | Master on/off switch for the client |
| `rate_limit_per_minute` | Requests/min cap (null = unlimited) |
| `allow_cloud_for_internal` | Whether `internal` jobs may use cloud models |

## Sensitivity tiers

Every job has an **effective sensitivity** that decides whether it may run on a
cloud model (Anthropic) or must stay local (Ollama).

| Tier | Cloud model allowed? |
|---|---|
| `public` | Yes |
| `internal` | Only if the client has `allow_cloud_for_internal` |
| `confidential` | **Never** — local models only |

This is the real data-egress control: `confidential` work physically cannot
leave to a third party, regardless of routing config.

### How the effective tier is chosen

The routing rule's `sensitivity` is a **floor**. The effective tier is the
*stricter* of the job's requested sensitivity and the rule's:

```
effective = stricter(job.sensitivity, rule.sensitivity)
```

So a client can **raise** strictness for a given job (e.g. mark a normally-
`internal` task `confidential`) but can never **relax** below the rule's floor.
With no matching rule, the floor defaults to `internal`.

### Enforcement

The chosen model (rule preferred → fallback, or an explicit `model` override on
the request) is checked against the effective tier:

- A **cloud** model on a `confidential` job, or on an `internal` job without
  `allow_cloud_for_internal`, is disallowed.
- If the preferred model is disallowed but the fallback is eligible, the
  fallback is promoted.
- If nothing is allowed (or an explicit override is disallowed), the request
  fails with `400` (`SensitivityViolation`).

Local models are always allowed. See [architecture.md](architecture.md) for the
full routing decision.

## Rate limiting

Per-client, keyed on `ClientApp.id` (not IP), using a Redis tumbling-window
counter that resets each minute:

- `rate_limit_per_minute = null` → unlimited.
- Over the limit → `429 Too Many Requests` with `Retry-After`,
  `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` headers.

The budget is shared across all of a client's request sources (same key
everywhere).

## Admin auth

A single shared secret, `CONDUCT_ADMIN_KEY` (generate with
`openssl rand -hex 32`), authenticates all admin endpoints as a Bearer token and
backs the UI login. The UI stores it in an HttpOnly `conduct_admin` cookie; set
`UI_COOKIE_SECURE=true` when serving the UI over HTTPS. There are no admin user
accounts — it's a single operator key.
