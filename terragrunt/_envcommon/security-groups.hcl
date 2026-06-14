# _envcommon/security-groups.hcl — all SGs, via the local module that wires
# them to each other (see modules/security-groups). Depends only on the VPC.

locals {
  region = read_terragrunt_config(find_in_parent_folders("region.hcl")).locals
  env    = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
}

terraform {
  source = "${dirname(find_in_parent_folders("root.hcl"))}/modules/security-groups"
}

dependency "vpc" {
  config_path = "../vpc"
  # mock_outputs let `terragrunt plan` / `validate` run before vpc is applied.
  mock_outputs = {
    vpc_id = "vpc-00000000000000000"
  }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}

inputs = {
  name_prefix = "conduct-${local.env.environment}"
  vpc_id      = dependency.vpc.outputs.vpc_id
}
