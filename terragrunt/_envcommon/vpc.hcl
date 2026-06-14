# _envcommon/vpc.hcl — shared VPC definition for every env/region.
# Each live/<env>/<region>/vpc/terragrunt.hcl includes this; the env/region
# specifics (CIDR, AZs, dev-vs-prod NAT strategy) come from region.hcl/env.hcl.

locals {
  region = read_terragrunt_config(find_in_parent_folders("region.hcl")).locals
  env    = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
}

terraform {
  source = "tfr:///terraform-aws-modules/vpc/aws//.?version=5.13.0"
}

inputs = {
  name = "conduct-${local.env.environment}-${local.region.aws_region}"
  cidr = local.region.vpc_cidr
  azs  = local.region.azs

  # Private subnets host the ECS tasks + data stores; public subnets host the
  # ALB + NAT. /20 private and /24 public carved from the /16, one per AZ.
  private_subnets = [for i, az in local.region.azs : cidrsubnet(local.region.vpc_cidr, 4, i)]
  public_subnets  = [for i, az in local.region.azs : cidrsubnet(local.region.vpc_cidr, 8, i + 48)]

  enable_nat_gateway = true
  # One NAT in dev (cheap, single point of failure is fine); one per AZ in prod
  # (no cross-AZ NAT dependency).
  single_nat_gateway     = local.env.environment != "prod"
  one_nat_gateway_per_az = local.env.environment == "prod"

  enable_dns_hostnames = true
  enable_dns_support   = true
}
