# _envcommon/vpc-peering.hcl — Conduct ⇄ Watchtower peering for this region.
# Exists only where BOTH stacks have a VPC (dev/us-west-2, prod/us-west-2,
# prod/us-east-2). Reads Watchtower's VPC from its remote state; see the
# vpc-peering module header for the full design.

locals {
  account = read_terragrunt_config(find_in_parent_folders("account.hcl")).locals
  env     = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
  region  = read_terragrunt_config(find_in_parent_folders("region.hcl")).locals

  # Watchtower uses the same live/<env>/<region>/vpc path, so its VPC state key
  # mirrors ours (path_relative_to_include would give the same here).
  peer_state_key = "live/${local.env.environment}/${local.region.aws_region}/vpc/terraform.tfstate"
}

terraform {
  source = "${dirname(find_in_parent_folders("root.hcl"))}/modules/vpc-peering"
}

dependency "vpc" {
  config_path = "../vpc"
  mock_outputs = {
    vpc_id                  = "vpc-00000000000000000"
    vpc_cidr_block          = "10.10.0.0/16"
    private_route_table_ids = ["rtb-a", "rtb-b", "rtb-c"]
  }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}

inputs = {
  name                      = "conduct-${local.env.environment}-to-watchtower-${local.region.aws_region}"
  requester_vpc_id          = dependency.vpc.outputs.vpc_id
  requester_cidr            = dependency.vpc.outputs.vpc_cidr_block
  requester_route_table_ids = dependency.vpc.outputs.private_route_table_ids

  peer_state_bucket = local.account.peer_state_bucket
  peer_state_key    = local.peer_state_key
  peer_state_region = local.account.peer_state_region
}
