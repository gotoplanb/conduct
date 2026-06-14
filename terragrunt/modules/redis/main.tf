# ============================================================================
# redis — ElastiCache for Redis (RQ broker + app cache)
# ============================================================================
# A small local module wrapping aws_elasticache_replication_group directly. We
# do this rather than use a registry module because the resource is simple and
# being explicit makes the dev (single node) vs prod (primary + replica,
# Multi-AZ, automatic failover) difference obvious and version-proof.

variable "name" { type = string }
variable "node_type" { type = string }
variable "num_cache_clusters" { type = number } # 1 = single node; >1 = primary + replicas
variable "engine_version" {
  type    = string
  default = "7.1"
}
variable "subnet_ids" { type = list(string) }
variable "security_group_ids" { type = list(string) }

resource "aws_elasticache_subnet_group" "this" {
  name       = var.name
  subnet_ids = var.subnet_ids
}

resource "aws_elasticache_replication_group" "this" {
  replication_group_id = var.name
  description          = "Conduct Redis (RQ broker + cache)"

  engine         = "redis"
  engine_version = var.engine_version
  node_type      = var.node_type
  port           = 6379

  num_cache_clusters         = var.num_cache_clusters
  automatic_failover_enabled = var.num_cache_clusters > 1
  multi_az_enabled           = var.num_cache_clusters > 1

  subnet_group_name  = aws_elasticache_subnet_group.this.name
  security_group_ids = var.security_group_ids

  at_rest_encryption_enabled = true
  # Transit encryption + AUTH token are OFF for simplicity: Redis is in private
  # subnets and only the app SG can reach it. To harden, set
  # transit_encryption_enabled = true + auth_token, and switch the app's
  # REDIS_URL scheme to rediss:// with the token.

  apply_immediately = true
}

output "primary_endpoint" {
  value = aws_elasticache_replication_group.this.primary_endpoint_address
}
output "port" {
  value = 6379
}
