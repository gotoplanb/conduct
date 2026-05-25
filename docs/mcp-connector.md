# Claude MCP connector

Conduct ships a remote [MCP](https://modelcontextprotocol.io) server so the
Claude apps (iOS / desktop / web) can list and create jobs as a **custom
connector**. It's an OAuth-protected Streamable-HTTP endpoint at `/mcp`.

## What it exposes

Four tools, all scoped to the connector's bound client app:

| Tool | Purpose |
|---|---|
| `list_task_types` | The task types this instance runs, with the model each routes to and its sensitivity floor |
| `list_jobs` | Your recent jobs (newest first), optionally filtered by status |
| `get_job` | A single job's status and result (response, tokens, cost) |
| `create_job` | Create a job — runs **asynchronously**; poll `get_job` for the result |

`create_job` always enqueues async (the natural pattern for a phone: create,
then ask Claude to fetch the result with `get_job`).

## How auth works

Conduct is its own OAuth 2.0 authorization server (authorization-code grant
with mandatory PKCE, plus refresh tokens). The pieces:

1. **A connector** = an OAuth client (`client_id` + `client_secret`) **bound to
   a Conduct client app**. Jobs created over MCP are attributed to that client
   app and inherit its rate limits and cloud permissions.
2. Claude does the OAuth dance against Conduct's discovery + `/oauth/authorize`
   + `/oauth/token` endpoints. You approve the consent screen by being logged
   into the Conduct UI (admin key).
3. The issued access token resolves to the client app on every `/mcp` call.

Secrets, codes, and tokens are stored only as SHA-256 hashes. **Deactivating a
connector immediately revokes all of its tokens** — the kill switch.

## Setup

### 1. Make Conduct publicly reachable over HTTPS

Claude must reach your instance, so it needs a public origin (a tunnel or
reverse proxy). Set:

- `CONDUCT_PUBLIC_URL` — the public origin (e.g. `https://conduct.example.app`).
  This becomes the OAuth issuer and the advertised MCP resource URL.
- `UI_COOKIE_SECURE=true` — once you're on HTTPS, so the admin session cookie
  used for consent is sent over the secure leg.

Restart the API after changing these. See [deployment.md](deployment.md) for
the tunnel setup.

### 2. Mint a connector

In the Conduct UI → **Connectors** → *New connector*:

- name it (e.g. `dave-ios`),
- bind it to a client app,
- set the redirect URI(s) (defaults to Claude's callback).

It shows the `client_id` and `client_secret` **once** — copy both.

### 3. Add it in Claude

Claude → Settings → Connectors → *Add custom connector*:

- Server URL: `https://<your-domain>/mcp`
- Under **Advanced settings**, paste the Client ID and Client Secret.

Connect → Claude sends you to Conduct to approve → the four tools appear.

## If you front Conduct with an auth proxy

If your tunnel/edge enforces its own auth (e.g. an ngrok traffic policy that
requires a Conduct `Bearer cdt_...` key), the OAuth/MCP flow will break, because
several steps legitimately arrive **without** that header:

- `/.well-known/oauth-*` — discovery, fetched with no auth
- `/oauth/authorize` (browser redirect) and `/oauth/token` (client-secret creds)
- the first `/mcp` probe, which must receive Conduct's `401 + WWW-Authenticate`
  challenge to start the flow
- `/ui/` and `/static/` — the login + consent screens

Allowlist those path prefixes at the edge so they bypass the coarse guard;
Conduct enforces its own auth on them. Keep the guard for `/jobs`, `/clients`,
etc.

## Troubleshooting

- **`401` on every request through the tunnel** → the edge proxy is blocking
  unauthenticated paths; see the allowlist note above.
- **Redirect downgrades to `http://`** → run uvicorn with `--proxy-headers
  --forwarded-allow-ips=*` so it honors `X-Forwarded-Proto` behind the TLS
  terminator (already set in the bundled compose/Dockerfile).
- **`421 Misdirected Request` on `/mcp`** → the MCP SDK's DNS-rebinding guard is
  rejecting the `Host`. Conduct allowlists the host from `CONDUCT_PUBLIC_URL`
  plus localhost; make sure `CONDUCT_PUBLIC_URL` matches the host Claude uses.
