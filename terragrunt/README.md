# Conduct on AWS — Terragrunt

This directory is the infrastructure-as-code for running **Conduct** on AWS. It
is written in [Terragrunt](https://terragrunt.gruntwork.io/) (a thin wrapper
around Terraform/OpenTofu) and follows the standard Terragrunt "live repo"
layout. This README is intentionally long: it's meant to be read top-to-bottom
as an explanation of *why* the tree is shaped the way it is, not just a command
cheat-sheet.

> **Status:** this is a skeleton meant to be read and refined. Nothing has been
> applied. Several values in `account.hcl` are placeholders you must fill in,
> and module versions should be re-pinned to the latest you trust before a real
> apply. No Terragrunt/Terraform is installed on the machine where this was
> written, so it has **not** been `hclfmt`/`validate`-checked — do that first
> (see [Validate before you trust it](#validate-before-you-trust-it)).

---

## 1. The mental model (read this first)

Terragrunt exists to solve one problem: **the same infrastructure, repeated
across environments and regions, without copy-pasting Terraform.** You describe
each *component* once, then "stamp" it into each environment/region with a tiny
file that says "use that description, here." The stamping files are nearly empty
because all the variation (sizes, CIDRs, counts) is read from the folder you're
standing in.

There are four kinds of files, and understanding them is 90% of understanding
the tree:

| File | Lives at | Answers the question |
|------|----------|----------------------|
| `root.hcl` | repo top of the tree | *Where does state live? How is the AWS provider configured?* (identical everywhere) |
| `account.hcl` / `env.hcl` / `region.hcl` | account / env / region levels | *What's different about THIS account / environment / region?* (just values) |
| `_envcommon/<component>.hcl` | shared | *What is component X — which module, which inputs?* (described once) |
| `live/.../<component>/terragrunt.hcl` | each leaf | *Create component X here.* (a 6-line include; the actual "unit") |

When you run a command in a leaf folder, Terragrunt walks **up** the directory
tree collecting `root.hcl`, `account.hcl`, `env.hcl`, `region.hcl`, merges them,
pulls in the matching `_envcommon` description, and hands the result to
Terraform. So `live/prod/us-east-2/rds/` and `live/dev/us-west-2/rds/` run the
*same* RDS description with *different* inputs — purely because of where they
sit in the tree.

---

## 2. Directory layout

```
terragrunt/
├── root.hcl                  # remote state (S3+DynamoDB) + generated AWS provider + common tags
├── account.hcl               # account id, state bucket/lock names, DNS zone   <-- EDIT THIS
│
├── _envcommon/               # each component described ONCE (module source + inputs + deps)
│   ├── vpc.hcl
│   ├── security-groups.hcl
│   ├── ecr.hcl
│   ├── secrets.hcl
│   ├── acm.hcl
│   ├── rds.hcl
│   ├── redis.hcl
│   ├── efs.hcl
│   ├── ecs-cluster.hcl
│   ├── alb.hcl
│   ├── dns.hcl
│   ├── ecs-service-api.hcl
│   └── ecs-service-worker.hcl
│
├── modules/                  # small LOCAL Terraform modules (where no good registry module fits)
│   ├── security-groups/      #   all SGs, cross-wired (avoids inter-unit dependency cycles)
│   ├── app-secrets/          #   Secrets Manager containers (values set out-of-band)
│   ├── redis/                #   ElastiCache replication group (explicit = clearer than the registry module)
│   └── route53-alias/        #   app A/ALIAS record, with multi-region latency routing
│
└── live/                     # the actual deployments — "stamps" of the components
    ├── dev/                  # dev: single region
    │   ├── env.hcl           #   dev sizes/counts                              <-- tune
    │   └── us-west-2/
    │       ├── region.hcl    #   dev us-west-2 CIDR/AZs
    │       └── <component>/terragrunt.hcl   (13 units)
    └── prod/                 # prod: MULTI-region (this is why we use Terragrunt)
        ├── env.hcl           #   prod sizes/counts (HA on)                     <-- tune
        ├── us-west-2/
        │   ├── region.hcl
        │   └── <component>/terragrunt.hcl   (13 units)
        └── us-east-2/
            ├── region.hcl
            └── <component>/terragrunt.hcl   (13 units)
```

Adding `us-east-1` to prod later is: copy a `region.hcl`, copy the 13 unit
folders (they're identical), pick a non-overlapping CIDR. That's the multi-region
payoff — region count is a folder operation, not a code change.

---

## 3. What gets deployed (the components)

All compute is **ECS Fargate** (serverless containers — no EC2 to patch). Per
environment/region:

| Component | Module | What it is |
|-----------|--------|------------|
| `vpc` | `terraform-aws-modules/vpc` | network: public subnets (ALB/NAT) + private subnets (everything else), across 3 AZs |
| `security-groups` | local | ALB / app / RDS / Redis / EFS SGs, wired so only the right thing can talk to the next thing |
| `ecr` | `terraform-aws-modules/ecr` | the container image repo (api + worker share one image) |
| `secrets` | local `app-secrets` | Secrets Manager *containers* for DATABASE_URL, admin key, secrets key, Anthropic key |
| `acm` | `terraform-aws-modules/acm` | TLS cert for the app hostname, DNS-validated |
| `rds` | `terraform-aws-modules/rds` | managed Postgres 16; RDS generates+stores its own master password |
| `redis` | local `redis` | ElastiCache for Redis (the RQ broker + cache) |
| `efs` | `terraform-aws-modules/efs` | shared filesystem for `/app/output` (api writes, worker writes, api serves) |
| `ecs-cluster` | `terraform-aws-modules/ecs//modules/cluster` | the Fargate cluster |
| `alb` | `terraform-aws-modules/alb` | public load balancer; :80→:443 redirect, :443→api |
| `dns` | local `route53-alias` | the public A record → ALB (latency-routed in prod) |
| `ecs-service-api` | `terraform-aws-modules/ecs//modules/service` | the FastAPI service behind the ALB |
| `ecs-service-worker` | `terraform-aws-modules/ecs//modules/service` | the RQ worker (no inbound) |
| `vpc-peering` | local | Conduct ⇄ Watchtower VPC peering + both-sides routes (cross-stack; see §8) |

### How this maps from `docker-compose.yml`

| compose service | becomes |
|-----------------|---------|
| `postgres` | RDS (managed) |
| `redis` | ElastiCache (managed) |
| `api` | `ecs-service-api` (Fargate, behind ALB) |
| `worker` | `ecs-service-worker` (Fargate) |
| `./output` bind mount | EFS access point mounted at `/app/output` in both tasks |
| `./voices` bind mount | **not handled** — see [Out of scope](#7-whats-deliberately-out-of-scope) |
| `host.docker.internal:11434` (Ollama) | **gone** — models come from Bedrock (see below) |
| `host.docker.internal:4317` (OTEL→Alloy) | Watchtower's OTLP endpoint (cross-stack; see §8) |

---

## 4. Dependencies & order of operations

Terragrunt reads `dependency` blocks to build a DAG and applies in the right
order automatically when you use `run-all`. The graph per region:

```
vpc
├── security-groups
│   ├── rds            (also needs vpc subnets)
│   ├── redis          (also needs vpc subnets)
│   ├── efs            (also needs vpc subnets)
│   └── alb            (also needs vpc public subnets + acm)
├── ecr
├── secrets
├── acm                (needs your Route53 zone)
└── ecs-cluster

alb ── dns

ecs-service-api    needs: ecs-cluster, vpc, security-groups, alb, ecr, redis, efs, secrets, rds
ecs-service-worker needs: ecs-cluster, vpc, security-groups,      ecr, redis, efs, secrets, rds
```

**Order of operations, first time:**

1. **Bootstrap** (once): fill in `account.hcl`; ensure AWS credentials for the
   target account; the S3 state bucket + DynamoDB lock table are created
   automatically by Terragrunt on the first run (it asks before creating).
2. **Build & push the image** to ECR — but ECR must exist first, so either
   apply the `ecr` unit first, or apply everything and let the services fail
   their first pull, then push and they recover. Recommended: apply `ecr`, push,
   then apply the rest. (See §6.)
3. **`run-all apply`** the region — Terragrunt walks the DAG above.
4. **Set the secret values** (DATABASE_URL etc.) — see §5. The services can't
   become healthy until DATABASE_URL is real.
5. **Run DB migrations** — `alembic upgrade head` against the new RDS. Easiest
   via a one-off ECS task (`aws ecs run-task` with the worker task def and an
   `alembic upgrade head` command override) so it runs inside the VPC.
6. **Seed** routing rules if desired (the app's `make seed` equivalent), again
   as a one-off task.

The `dependency` blocks include `mock_outputs` gated to `plan`/`validate`/`init`
so you can `terragrunt run-all plan` the whole tree before anything exists — the
mocks satisfy references that aren't created yet. Real applies use real outputs.

---

## 5. Secrets & environment variables you set yourself

Terraform should never hold real secrets (they'd land in state). So the
`secrets` unit creates Secrets Manager *containers* with a `REPLACE_ME`
placeholder and `ignore_changes` on the value — **you set the real values once,
out of band.** The ECS tasks read them at startup via `valueFrom`.

After `rds` and `secrets` are applied, set each value:

```bash
# DATABASE_URL — compose it from the RDS endpoint + the RDS-managed password.
# The password lives in the RDS-created secret; fetch it, then write the URL.
RDS_ENDPOINT=$(aws rds describe-db-instances \
  --db-instance-identifier conduct-prod \
  --query 'DBInstances[0].Endpoint.Address' --output text)
RDS_SECRET_ARN=$(aws rds describe-db-instances \
  --db-instance-identifier conduct-prod \
  --query 'DBInstances[0].MasterUserSecret.SecretArn' --output text)
RDS_PASSWORD=$(aws secretsmanager get-secret-value --secret-id "$RDS_SECRET_ARN" \
  --query SecretString --output text | jq -r .password)

aws secretsmanager put-secret-value --secret-id conduct/prod/database-url \
  --secret-string "postgresql+asyncpg://conduct:${RDS_PASSWORD}@${RDS_ENDPOINT}:5432/conduct"

# Admin + Fernet keys (same generators as .env.example)
aws secretsmanager put-secret-value --secret-id conduct/prod/admin-key \
  --secret-string "$(openssl rand -hex 32)"
aws secretsmanager put-secret-value --secret-id conduct/prod/secrets-key \
  --secret-string "$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"

# Anthropic key — only if you use the direct Anthropic provider. With Bedrock
# you don't need it; leave the REPLACE_ME placeholder.
```

> Rotating `secrets-key` orphans any already-encrypted per-client secrets in the
> DB (same warning as local `.env`). Back it up.

**Non-secret env vars** are set for you by the Terragrunt, sourced from the
hierarchy — you don't touch these per-deploy except where noted:

| Env var | Where it comes from |
|---------|---------------------|
| `REDIS_URL` | composed from the `redis` unit's endpoint |
| `DEFAULT_MODEL` / `DEFAULT_SENSITIVE_MODEL` | `env.hcl` (Bedrock model ids) |
| `CONDUCT_PUBLIC_URL` | `https://<fqdn>` from `account.hcl.domain_base` (or `public_url_override`) |
| `AWS_REGION` | the region — Bedrock client uses it + the task role |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `env.hcl.otlp_endpoint` (Watchtower; blank until §8) |
| `GRAFANA_BASE_URL` | `env.hcl.grafana_url` (Watchtower; blank until §8) |
| `TTS_VOICES_DIR` / `TTS_OUTPUT_DIR` | fixed (`/app/voices`, `/app/output`) |

The **one** value you bump every deploy is `env.hcl.image_tag` (the git SHA you
pushed). ECR is immutable-tagged, so `latest` won't exist — set the SHA, then
apply the two service units.

---

## 6. Prerequisites & first deploy

**Tooling:** [Terragrunt](https://terragrunt.gruntwork.io/docs/getting-started/install/),
Terraform **or** OpenTofu, the AWS CLI (authenticated to the target account),
Docker with `buildx`, and `jq`. (None are installed on this machine yet.)

**Edit `account.hcl`:** `account_id`, a globally-unique `state_bucket`, and —
if you have a domain — `hosted_zone_name`, `hosted_zone_id`, `domain_base`.

**Build & push the image** (ARM64, because the task defs request Graviton —
cheaper; switch the `runtime_platform` to `X86_64` if you'd rather push amd64):

```bash
cd /Users/dave/conduct
SHA=$(git rev-parse HEAD)
ACCOUNT=111122223333; REGION=us-west-2
REPO="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/conduct-prod"

# apply ecr first so the repo exists
cd terragrunt/live/prod/us-west-2/ecr && terragrunt apply && cd -

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
docker buildx build --platform linux/arm64 --build-arg GIT_SHA=$SHA -t "$REPO:$SHA" --push .
# repeat the push for each region's repo (or set up ECR replication)
```

Set `image_tag = "<SHA>"` in `live/prod/env.hcl`, then:

```bash
cd terragrunt/live/prod/us-west-2
terragrunt run-all plan      # review everything
terragrunt run-all apply     # create it (respects the dependency DAG)
```

Then do the §5 secret values + migrations, and repeat the `run-all` in
`us-east-2`. dev is the same flow under `live/dev/us-west-2`.

> **Per-unit vs run-all:** you can always `cd` into a single unit and
> `terragrunt apply` just that one (great for iterating on, say, the api
> service). `run-all` is for doing a whole region at once.

---

## 7. What's deliberately out of scope

This deployment runs **Conduct's text-dispatch core on Bedrock**. Intentionally
NOT included, because they need GPUs / Apple hardware that Fargate doesn't have:

- **Local Ollama models** — replaced by Bedrock (`DEFAULT_MODEL` is a Bedrock id;
  the task role grants `bedrock:InvokeModel`). Bedrock uses the task's IAM role,
  so there's no API key to manage.
- **Media tasks** (ComfyUI image/video, ACE-Step audio) — these call GPU servers
  that only run on your Mac. Routing a media task in this deployment will fail at
  dispatch. Phase 2 options: GPU EC2 (g5/g6) running Ollama/ComfyUI, or a private
  tunnel back to on-prem.
- **TTS/Piper voices** — the `./voices` files aren't in the image. If you want
  TTS in-cloud, bake the voice files into the image or stage them on the EFS
  filesystem, then they'll be at `/app/voices`.

If/when you want any of these, they're additive — they don't change what's here.

---

## 8. Cross-stack: talking to Watchtower

Conduct sends OpenTelemetry to Watchtower's **Alloy** and deep-links job pages
to its **Grafana**. Those live in the Watchtower Terragrunt (separate repo /
stack). The decoupling pattern:

1. **Networking:** the `vpc-peering` unit (one per region where both stacks
   exist: dev/us-west-2, prod/us-west-2, prod/us-east-2) creates the
   Conduct⇄Watchtower peering connection and the routes on *both* VPCs. Conduct
   is the requester; it reads Watchtower's VPC outputs from **Watchtower's
   remote state** (`account.hcl.peer_state_bucket` + a per-region key) rather
   than a cross-repo file path, so the stacks stay decoupled. CIDRs are
   pre-planned non-overlapping (Conduct 10.1x/10.2x, Watchtower 10.4x/10.5x).
   Conduct then reaches Alloy over the peering link via Watchtower's internal
   NLB. **Ordering:** Watchtower's `vpc` must be applied *before* Conduct's
   `vpc-peering` (the peering reads its state) — so the cross-stack sequence is:
   Watchtower vpc → Conduct vpc-peering. The unit also writes routes into
   Watchtower's route tables (the "consumer owns the link" choice); they don't
   conflict with Watchtower's VPC module.
2. **Discovery:** Watchtower publishes its OTLP endpoint + Grafana URL to SSM
   Parameter Store (e.g. `/watchtower/<env>/<region>/otlp-endpoint`). For now
   you copy those into `env.hcl` (`otlp_endpoint`, `grafana_url`); a later
   iteration can have Conduct read the SSM params directly via a data source.
3. **Order:** bring Watchtower up first (so the endpoints exist), then set
   Conduct's two values and apply the services. Until then, leave them blank —
   the app runs fine without telemetry wired.

This is described from Watchtower's side in its own `terragrunt/README.md`.

---

## 9. Validate before you trust it

No IaC tooling was available where this was generated, so run these once you're
set up — expect to fix small things (module input/output names drift between
module versions):

```bash
cd terragrunt
terragrunt hcl fmt                      # format all .hcl
cd live/dev/us-west-2
terragrunt run-all validate             # provider/module wiring
terragrunt run-all plan                 # the real proof (uses mock_outputs for unbuilt deps)
```

Specifically re-verify, against the versions you pin:
- the **ECS service module** input names (`container_definitions`, `volume`,
  `task_exec_secret_arns`, `tasks_iam_role_statements`, `load_balancer`) — these
  have evolved across v5.x.
- the **ALB module** v9 `listeners`/`target_groups` shape.
- the **EFS module** `mount_targets` / `access_points` output shape.
Pin every `?version=` in `_envcommon/*.hcl` to the latest you trust.

---

## 10. Cost & teardown

- **Always-on baseline** (per region): NAT gateway(s), RDS, ElastiCache, the
  ALB, and the Fargate tasks are the cost floor. dev uses a single NAT + single
  small instances; prod uses NAT-per-AZ + Multi-AZ. NAT gateways and RDS are
  usually the biggest line items — turning dev off when unused saves the most.
- **Teardown:** `terragrunt run-all destroy` in a region. Order is reversed
  automatically. Note: prod RDS has `deletion_protection = true` and the ALB has
  `enable_deletion_protection = true` — you must disable those (or
  `--terragrunt-no-...`) to destroy prod, on purpose.

---

## 11. Design decisions (the "why", for the record)

- **ECS Fargate over EKS:** fewest moving parts for a handful of stateless
  containers; maps 1:1 from compose; no cluster/IRSA/addons to learn first.
- **Single AWS account:** simplest for a first deploy; the folder layout is
  already multi-account-shaped, so splitting prod into its own account later is
  mechanical.
- **Bedrock-only models:** no GPU needed, IAM-native, and Conduct already
  supports it. Keeps the cloud footprint small.
- **EFS for `/app/output`:** zero app changes vs the compose bind-mount. The
  cleaner end state is S3 + presigned URLs; deferred on purpose.
- **Local module for security groups:** the SGs reference each other; defining
  them together removes inter-unit dependency cycles and is easier to read than
  five `security-group` units.
- **Multi-region = independent regional stacks + latency DNS:** each region is a
  full, self-contained deployment; Route53 latency records send users to the
  nearest one (active-active). Hard failover (Route53 health checks) is a
  documented add-on, not built yet.
```
