# ============================================================================
# app-secrets — Secrets Manager containers for Conduct's sensitive config
# ============================================================================
# We create the secret *containers* (and a REPLACE_ME placeholder version so
# the ECS task's valueFrom references resolve on first boot) but never put the
# real values in Terraform — they'd land in state. You set the real values
# out-of-band once (console or CLI); see terragrunt/README.md. The
# ignore_changes on the version means Terraform won't clobber what you set.

variable "name_prefix" { type = string } # e.g. "conduct/dev"
variable "secret_names" {
  type = list(string)
  # database-url      : full asyncpg URL incl. the RDS-managed password
  # admin-key         : CONDUCT_ADMIN_KEY
  # secrets-key       : CONDUCT_SECRETS_KEY (Fernet master key)
  # anthropic-api-key : optional; only if you use the Anthropic direct provider
  #                     instead of (or alongside) Bedrock
  default = ["database-url", "admin-key", "secrets-key", "anthropic-api-key"]
}
variable "recovery_window_days" {
  type    = number
  default = 7
}

resource "aws_secretsmanager_secret" "this" {
  for_each                = toset(var.secret_names)
  name                    = "${var.name_prefix}/${each.value}"
  description             = "Conduct ${each.value} — value set out-of-band"
  recovery_window_in_days = var.recovery_window_days
}

resource "aws_secretsmanager_secret_version" "placeholder" {
  for_each      = aws_secretsmanager_secret.this
  secret_id     = each.value.id
  secret_string = "REPLACE_ME"
  lifecycle {
    ignore_changes = [secret_string]
  }
}
