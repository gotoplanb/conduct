# ============================================================================
# Root Terragrunt configuration
# ============================================================================
# Every unit (each leaf terragrunt.hcl under live/) pulls this in with:
#
#   include "root" { path = find_in_parent_folders("root.hcl") }
#
# It centralizes the three things that MUST be identical everywhere: where
# state lives, how the AWS provider is configured, and the tags stamped on
# every resource. Nothing here is region- or env-specific on its own — it
# reads those from account.hcl / env.hcl / region.hcl, which sit at the right
# levels of the folder tree.
#
# Note: the root file is named root.hcl (not terragrunt.hcl) and is located
# via find_in_parent_folders("root.hcl"). Recent Terragrunt deprecates the
# implicit terragrunt.hcl-as-root behavior, so we name it explicitly.

locals {
  account_vars = read_terragrunt_config(find_in_parent_folders("account.hcl"))
  env_vars     = read_terragrunt_config(find_in_parent_folders("env.hcl"))
  region_vars  = read_terragrunt_config(find_in_parent_folders("region.hcl"))

  account_id  = local.account_vars.locals.account_id
  aws_region  = local.region_vars.locals.aws_region
  environment = local.env_vars.locals.environment

  state_bucket = local.account_vars.locals.state_bucket
  state_region = local.account_vars.locals.state_region
  lock_table   = local.account_vars.locals.lock_table

  # Applied to every resource that supports tagging (via provider default_tags).
  common_tags = {
    Project     = "conduct"
    Environment = local.environment
    Region      = local.aws_region
    ManagedBy   = "terragrunt"
  }
}

# ----------------------------------------------------------------------------
# Remote state
# ----------------------------------------------------------------------------
# One S3 bucket + one DynamoDB lock table for the whole account. The state KEY
# is derived from the unit's path relative to this root, so
# live/prod/us-east-2/rds and live/dev/us-west-2/rds get distinct state files
# automatically — no manual key bookkeeping, and adding a region can never
# collide with an existing one. Terragrunt creates the bucket + table on the
# first run if they don't already exist.
remote_state {
  backend = "s3"
  generate = {
    path      = "backend.tf"
    if_exists = "overwrite_terragrunt"
  }
  config = {
    bucket         = local.state_bucket
    key            = "${path_relative_to_include()}/terraform.tfstate"
    region         = local.state_region
    encrypt        = true
    dynamodb_table = local.lock_table
  }
}

# ----------------------------------------------------------------------------
# AWS provider
# ----------------------------------------------------------------------------
# Generated into every unit. The region comes from region.hcl, so the SAME
# unit definition deployed under us-west-2/ vs us-east-2/ targets the correct
# region with zero duplication. allowed_account_ids is a guardrail: a
# misconfigured credential pointing at the wrong account fails fast.
generate "provider" {
  path      = "provider.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<EOF
provider "aws" {
  region              = "${local.aws_region}"
  allowed_account_ids = ["${local.account_id}"]

  default_tags {
    tags = ${jsonencode(local.common_tags)}
  }
}
EOF
}

# ----------------------------------------------------------------------------
# Common inputs
# ----------------------------------------------------------------------------
# Merged into every unit. Individual units (and _envcommon files) can read
# these or override them. We expose the resolved account/env/region locals so
# component configs don't each have to re-read the same files.
inputs = merge(
  local.account_vars.locals,
  local.env_vars.locals,
  local.region_vars.locals,
  {
    common_tags = local.common_tags
  },
)
