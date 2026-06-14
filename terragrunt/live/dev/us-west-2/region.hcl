# dev / us-west-2 region variables. See prod/us-west-2/region.hcl for the CIDR plan.
locals {
  aws_region = "us-west-2"
  azs        = ["us-west-2a", "us-west-2b", "us-west-2c"]
  vpc_cidr   = "10.20.0.0/16"
}
