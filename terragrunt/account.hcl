# ============================================================================
# Account-wide constants (single-account topology)
# ============================================================================
# We deploy every environment and region into ONE AWS account. Isolation
# between dev and prod comes from naming, tags, and separate state keys rather
# than separate accounts. The folder layout is identical to what you'd use for
# a multi-account setup, so splitting prod into its own account later is a
# mechanical change (move this file down a level and give each account its own
# account.hcl) rather than a restructure.
#
# Everything here is a placeholder you must fill in before the first apply.

locals {
  # Your 12-digit AWS account id. The provider's allowed_account_ids guardrail
  # checks against this.
  account_id   = "111122223333" # TODO: replace with your AWS account id
  account_name = "conduct"

  # ---- Remote state backend -------------------------------------------------
  # Created automatically by Terragrunt on the first run. The bucket is global
  # across all regions (state for us-east-2 resources still lives here); it
  # just physically resides in state_region. S3 bucket names are GLOBALLY
  # unique, so the account-id suffix keeps it yours.
  state_region = "us-west-2"
  state_bucket = "conduct-tfstate-111122223333" # TODO: make this unique to you
  lock_table   = "conduct-tflocks"

  # ---- Public DNS / TLS -----------------------------------------------------
  # A Route53 PUBLIC hosted zone you already own. Used to issue the ACM cert
  # and create the app's A/ALIAS record.
  #
  # Leave both blank to skip ACM + DNS entirely and reach the app over the raw
  # ALB DNS name on HTTP — fine for a first smoke test, but Conduct's OAuth
  # issuer + MCP resource URL need real HTTPS, so set these before real use.
  hosted_zone_name = "" # e.g. "example.com"  (the zone)
  hosted_zone_id   = "" # e.g. "Z0123456789ABCDEFGHIJ" (find: aws route53 list-hosted-zones)
  domain_base      = "" # e.g. "conduct.example.com" (env subdomains derived below)

  # ---- Cross-stack: Watchtower's Terraform state -----------------------------
  # The vpc-peering unit reads Watchtower's VPC outputs from its remote state to
  # build the Conduct⇄Watchtower peering. Must match watchtower/terragrunt/
  # account.hcl's state_bucket + state_region. The state KEY is derived
  # per-region inside _envcommon/vpc-peering.hcl (same live/ path in both repos).
  peer_state_bucket = "watchtower-tfstate-111122223333" # TODO: match Watchtower's
  peer_state_region = "us-west-2"
}
