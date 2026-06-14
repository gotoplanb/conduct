# ============================================================================
# prod environment variables
# ============================================================================
# prod is multi-region: this env folder contains BOTH us-west-2/ and
# us-east-2/. Every value here applies identically to both regions; anything
# that must differ per region lives in that region's region.hcl.

locals {
  environment = "prod"

  # Data tier — small Graviton, Multi-AZ for failover. Bump instance classes
  # as real load shows up; these are deliberately modest starting points.
  rds_instance_class = "db.t4g.small"
  rds_multi_az       = true
  redis_node_type    = "cache.t4g.small"
  redis_num_nodes    = 2 # primary + 1 replica

  # App tier — two tasks each for rolling deploys + AZ spread.
  api_desired_count    = 2
  api_cpu              = 512
  api_memory           = 1024
  worker_desired_count = 2
  worker_cpu           = 512
  worker_memory        = 1024

  public_url_override = ""

  # See dev/env.hcl for what each of these does.
  image_tag = "latest"

  default_model           = "us.anthropic.claude-sonnet-4-6"
  default_sensitive_model = "us.anthropic.claude-sonnet-4-6"

  otlp_endpoint = ""
  grafana_url   = ""
}
