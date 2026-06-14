# prod / us-west-2 region variables.
#
# CIDR PLAN (keep every VPC non-overlapping so VPC peering — e.g. to
# Watchtower's Alloy — is always possible):
#   conduct  dev  us-west-2  10.20.0.0/16
#   conduct  prod us-west-2  10.10.0.0/16   <-- this file
#   conduct  prod us-east-2  10.11.0.0/16
#   (Watchtower uses 10.40.0.0/16+ — see its region.hcl files.)
locals {
  aws_region = "us-west-2"
  azs        = ["us-west-2a", "us-west-2b", "us-west-2c"]
  vpc_cidr   = "10.10.0.0/16"
}
