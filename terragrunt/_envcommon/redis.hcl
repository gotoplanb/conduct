# _envcommon/redis.hcl — ElastiCache for Redis via the local redis module.

locals {
  env = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
}

terraform {
  source = "${dirname(find_in_parent_folders("root.hcl"))}/modules/redis"
}

dependency "vpc" {
  config_path                             = "../vpc"
  mock_outputs                            = { private_subnets = ["subnet-aaa", "subnet-bbb", "subnet-ccc"] }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}

dependency "sg" {
  config_path                             = "../security-groups"
  mock_outputs                            = { redis_sg_id = "sg-redis000" }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}

inputs = {
  name               = "conduct-${local.env.environment}"
  node_type          = local.env.redis_node_type
  num_cache_clusters = local.env.redis_num_nodes
  subnet_ids         = dependency.vpc.outputs.private_subnets
  security_group_ids = [dependency.sg.outputs.redis_sg_id]
}
