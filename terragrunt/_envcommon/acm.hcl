# _envcommon/acm.hcl — TLS certificate for the app's hostname, DNS-validated
# against your Route53 zone. ACM certs are REGIONAL (an ALB needs a cert in its
# own region), so there's an acm unit per region.
#
# Hostname convention: prod -> "<domain_base>", non-prod -> "<env>.<domain_base>".
#
# IF account.hcl.domain_base / hosted_zone_id are blank, do NOT run this unit
# (or the alb's HTTPS listener). See README "No domain yet?".

locals {
  account = read_terragrunt_config(find_in_parent_folders("account.hcl")).locals
  env     = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals

  fqdn = local.env.environment == "prod" ? local.account.domain_base : "${local.env.environment}.${local.account.domain_base}"
}

terraform {
  source = "tfr:///terraform-aws-modules/acm/aws//.?version=5.1.1"
}

inputs = {
  domain_name = local.fqdn
  zone_id     = local.account.hosted_zone_id

  validation_method      = "DNS"
  create_route53_records = true
  wait_for_validation    = true
}
