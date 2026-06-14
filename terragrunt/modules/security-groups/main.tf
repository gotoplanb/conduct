# ============================================================================
# security-groups — all of Conduct's SGs in one place
# ============================================================================
# Why a local module instead of N units of terraform-aws-modules/security-group?
# Because the rules reference EACH OTHER (the RDS SG allows the app SG; the app
# SG allows the ALB SG). Defining them together lets every cross-reference be a
# plain resource attribute with no inter-unit dependency cycle. Each consuming
# unit (alb, rds, redis, efs, the ECS services) just takes the relevant SG id
# from this module's outputs and is told NOT to create its own.
#
# Topology:
#   internet --443/80--> [alb_sg] --8000--> [app_sg] --5432--> [rds_sg]
#                                                    --6379--> [redis_sg]
#                                                    --2049--> [efs_sg]

variable "name_prefix" { type = string }
variable "vpc_id" { type = string }
variable "api_container_port" {
  type    = number
  default = 8000
}

# ---- ALB: public ingress on 80/443 ----------------------------------------
resource "aws_security_group" "alb" {
  name_prefix = "${var.name_prefix}-alb-"
  description = "Conduct ALB — public HTTP/HTTPS ingress"
  vpc_id      = var.vpc_id
  lifecycle { create_before_destroy = true }
}

resource "aws_vpc_security_group_ingress_rule" "alb_https" {
  security_group_id = aws_security_group.alb.id
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 443
  to_port           = 443
  ip_protocol       = "tcp"
  description       = "HTTPS from anywhere"
}

resource "aws_vpc_security_group_ingress_rule" "alb_http" {
  security_group_id = aws_security_group.alb.id
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 80
  to_port           = 80
  ip_protocol       = "tcp"
  description       = "HTTP (redirected to HTTPS at the listener)"
}

resource "aws_vpc_security_group_egress_rule" "alb_all" {
  security_group_id = aws_security_group.alb.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
  description       = "ALB to targets"
}

# ---- App (ECS tasks: api + worker) ----------------------------------------
resource "aws_security_group" "app" {
  name_prefix = "${var.name_prefix}-app-"
  description = "Conduct ECS tasks (api + worker)"
  vpc_id      = var.vpc_id
  lifecycle { create_before_destroy = true }
}

# Only the ALB may reach the api's container port. The worker has no inbound.
resource "aws_vpc_security_group_ingress_rule" "app_from_alb" {
  security_group_id            = aws_security_group.app.id
  referenced_security_group_id = aws_security_group.alb.id
  from_port                    = var.api_container_port
  to_port                      = var.api_container_port
  ip_protocol                  = "tcp"
  description                  = "ALB to api container"
}

# Egress all — tasks call out to RDS, Redis, EFS, Bedrock, and the Watchtower
# OTLP endpoint. Tighten later with PrivateLink/endpoint-specific rules.
resource "aws_vpc_security_group_egress_rule" "app_all" {
  security_group_id = aws_security_group.app.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
  description       = "App egress (data stores, Bedrock, OTLP)"
}

# ---- Data stores: ingress only from the app SG ----------------------------
locals {
  data_ports = {
    rds   = 5432
    redis = 6379
    efs   = 2049
  }
}

resource "aws_security_group" "data" {
  for_each    = local.data_ports
  name_prefix = "${var.name_prefix}-${each.key}-"
  description = "Conduct ${each.key} — app-only ingress"
  vpc_id      = var.vpc_id
  lifecycle { create_before_destroy = true }
}

resource "aws_vpc_security_group_ingress_rule" "data_from_app" {
  for_each                     = local.data_ports
  security_group_id            = aws_security_group.data[each.key].id
  referenced_security_group_id = aws_security_group.app.id
  from_port                    = each.value
  to_port                      = each.value
  ip_protocol                  = "tcp"
  description                  = "App to ${each.key}"
}

resource "aws_vpc_security_group_egress_rule" "data_all" {
  for_each          = local.data_ports
  security_group_id = aws_security_group.data[each.key].id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}
