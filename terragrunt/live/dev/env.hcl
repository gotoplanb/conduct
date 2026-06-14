# ============================================================================
# dev environment variables
# ============================================================================
# Read by root.hcl and by every _envcommon component. Holds the knobs that
# differ by environment (size, counts, HA). dev is intentionally small and
# single-region (only a us-west-2/ folder exists under here).

locals {
  environment = "dev"

  # Data tier — smallest burstable Graviton instances, single-AZ.
  rds_instance_class = "db.t4g.micro"
  rds_multi_az       = false
  redis_node_type    = "cache.t4g.micro"
  redis_num_nodes    = 1

  # App tier — one task each, minimal Fargate sizing (CPU in vCPU units,
  # memory in MiB; 256/512 is the smallest valid Fargate combo).
  api_desired_count    = 1
  api_cpu              = 256
  api_memory           = 512
  worker_desired_count = 1
  worker_cpu           = 256
  worker_memory        = 512

  # CONDUCT_PUBLIC_URL the app advertises (OAuth issuer / MCP resource). If
  # account.hcl.domain_base is set, units build "<env>.<domain_base>"; this is
  # the explicit override when you want something else.
  public_url_override = ""

  # Container image tag to run. ECR repos are IMMUTABLE and tagged by git SHA,
  # so bump this to the SHA you pushed, then `terragrunt apply` the services.
  # (A CI pipeline would set this; "latest" won't exist in an immutable repo.)
  image_tag = "latest"

  # Bedrock model ids the router defaults to (Models = Bedrock-only on cloud).
  default_model           = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
  default_sensitive_model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

  # Cross-stack (Watchtower). Leave blank until Watchtower is up + VPC-peered,
  # then set to its internal OTLP NLB endpoint and Grafana URL. Blank = telemetry
  # env vars omitted (app still runs).
  otlp_endpoint = "" # e.g. "http://otlp.watchtower.internal:4317"
  grafana_url   = "" # e.g. "https://grafana.example.com"
}
