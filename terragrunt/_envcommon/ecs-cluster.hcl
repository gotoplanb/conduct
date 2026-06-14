# _envcommon/ecs-cluster.hcl — the Fargate cluster that hosts the api + worker
# services. Container Insights on for CloudWatch metrics. No EC2 capacity —
# pure Fargate (FARGATE + FARGATE_SPOT capacity providers).

locals {
  env = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
}

terraform {
  source = "tfr:///terraform-aws-modules/ecs/aws//modules/cluster?version=5.12.0"
}

inputs = {
  cluster_name = "conduct-${local.env.environment}"

  cluster_settings = [{
    name  = "containerInsights"
    value = "enabled"
  }]

  fargate_capacity_providers = {
    FARGATE      = { default_capacity_provider_strategy = { weight = 1, base = 1 } }
    FARGATE_SPOT = { default_capacity_provider_strategy = { weight = 0 } }
  }
}
