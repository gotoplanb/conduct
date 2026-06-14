# _envcommon/secrets.hcl — Secrets Manager containers for app secrets.
# Values are set out-of-band (see terragrunt/README.md "Secrets you set
# yourself"). No upstream dependencies.

locals {
  env = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
}

terraform {
  source = "${dirname(find_in_parent_folders("root.hcl"))}/modules/app-secrets"
}

inputs = {
  name_prefix = "conduct/${local.env.environment}"
}
