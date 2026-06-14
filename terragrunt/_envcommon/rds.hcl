# _envcommon/rds.hcl — managed Postgres 16 for Conduct.
#
# manage_master_user_password = true → RDS generates the master password and
# stores it in its OWN Secrets Manager secret. We never see it in state. You
# reference that secret (plus db_instance_endpoint) when you compose the
# `database-url` app secret (see README). Multi-AZ in prod, single-AZ in dev.

locals {
  region = read_terragrunt_config(find_in_parent_folders("region.hcl")).locals
  env    = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
}

terraform {
  source = "tfr:///terraform-aws-modules/rds/aws//.?version=6.10.0"
}

dependency "vpc" {
  config_path = "../vpc"
  mock_outputs = {
    private_subnets = ["subnet-aaa", "subnet-bbb", "subnet-ccc"]
  }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}

dependency "sg" {
  config_path                             = "../security-groups"
  mock_outputs                            = { rds_sg_id = "sg-rds0000" }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}

inputs = {
  identifier = "conduct-${local.env.environment}"

  engine               = "postgres"
  engine_version       = "16"
  family               = "postgres16"
  major_engine_version = "16"
  instance_class       = local.env.rds_instance_class

  allocated_storage     = 20
  max_allocated_storage = 100 # storage autoscaling ceiling
  storage_encrypted     = true

  db_name  = "conduct"
  username = "conduct"
  port     = 5432

  manage_master_user_password = true
  multi_az                    = local.env.rds_multi_az

  create_db_subnet_group = true
  subnet_ids             = dependency.vpc.outputs.private_subnets
  vpc_security_group_ids = [dependency.sg.outputs.rds_sg_id]

  backup_retention_period = local.env.environment == "prod" ? 14 : 1
  deletion_protection     = local.env.environment == "prod"
  skip_final_snapshot     = local.env.environment != "prod"

  performance_insights_enabled = true
}
