# Map of logical name -> secret ARN, consumed by the ECS services to wire
# container `secrets` (valueFrom) entries.
output "secret_arns" {
  value = { for k, s in aws_secretsmanager_secret.this : k => s.arn }
}
