# ============================================================================
# route53-alias — point the app hostname at a regional ALB
# ============================================================================
# In multi-region prod, BOTH regions want an A record for the SAME hostname.
# A plain alias would collide, so when enable_latency_routing = true we attach
# a LATENCY routing policy keyed by region + a per-region set_identifier:
# Route53 then answers each client with the ALB in the AWS region closest to
# them (active-active). In single-region dev we create a simple alias.
#
# Failover note: latency routing + the ALB target-group health check gives you
# soft failover (an unhealthy region's ALB still resolves but returns 5xx). For
# hard DNS failover, add Route53 health checks + failover records — see README.

variable "zone_id" { type = string }
variable "name" { type = string } # fqdn, e.g. conduct.example.com
variable "alb_dns_name" { type = string }
variable "alb_zone_id" { type = string }
variable "enable_latency_routing" {
  type    = bool
  default = false
}
variable "region" { type = string } # used as set_identifier for latency records

resource "aws_route53_record" "this" {
  zone_id = var.zone_id
  name    = var.name
  type    = "A"

  alias {
    name                   = var.alb_dns_name
    zone_id                = var.alb_zone_id
    evaluate_target_health = true
  }

  # Only set when doing multi-region latency routing. For a single record both
  # must be absent, hence the dynamic blocks.
  dynamic "latency_routing_policy" {
    for_each = var.enable_latency_routing ? [1] : []
    content {
      region = var.region
    }
  }
  set_identifier = var.enable_latency_routing ? var.region : null
}

output "fqdn" {
  value = aws_route53_record.this.fqdn
}
