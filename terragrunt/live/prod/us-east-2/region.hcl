# prod / us-east-2 region variables. See us-west-2/region.hcl for the CIDR plan.
locals {
  aws_region = "us-east-2"
  azs        = ["us-east-2a", "us-east-2b", "us-east-2c"]
  vpc_cidr   = "10.11.0.0/16"
}
