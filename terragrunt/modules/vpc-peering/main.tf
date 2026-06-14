# ============================================================================
# vpc-peering — Conduct ⇄ Watchtower (intra-region, same account)
# ============================================================================
# Conduct is the requester (it reaches into Watchtower for OTLP). This module:
#   1. reads the PEER (Watchtower) VPC's outputs from its remote state,
#   2. creates the peering connection (auto_accept — valid for same-account,
#      same-region peering, so no separate accepter resource), and
#   3. adds routes to BOTH VPCs' private route tables (Conduct→Watchtower CIDR
#      and Watchtower→Conduct CIDR) so traffic actually flows.
#
# Reading the peer via terraform_remote_state (not a cross-repo file path)
# keeps the two stacks decoupled: this only needs Watchtower's state bucket +
# key, which we already know. NOTE: Watchtower's VPC must be applied before this
# unit (its state has to exist + have outputs).

variable "name" { type = string }

# Requester side = Conduct's own VPC (passed in from the local vpc dependency).
variable "requester_vpc_id" { type = string }
variable "requester_cidr" { type = string }
variable "requester_route_table_ids" { type = list(string) }

# Where Watchtower's VPC state lives.
variable "peer_state_bucket" { type = string }
variable "peer_state_key" { type = string }
variable "peer_state_region" { type = string }

data "terraform_remote_state" "peer" {
  backend = "s3"
  config = {
    bucket = var.peer_state_bucket
    key    = var.peer_state_key
    region = var.peer_state_region
  }
}

locals {
  accepter_vpc_id          = data.terraform_remote_state.peer.outputs.vpc_id
  accepter_cidr            = data.terraform_remote_state.peer.outputs.vpc_cidr_block
  accepter_route_table_ids = data.terraform_remote_state.peer.outputs.private_route_table_ids
}

resource "aws_vpc_peering_connection" "this" {
  vpc_id      = var.requester_vpc_id
  peer_vpc_id = local.accepter_vpc_id
  auto_accept = true
  tags        = { Name = var.name }
}

# Conduct private subnets -> Watchtower CIDR
resource "aws_route" "requester_to_accepter" {
  count                     = length(var.requester_route_table_ids)
  route_table_id            = var.requester_route_table_ids[count.index]
  destination_cidr_block    = local.accepter_cidr
  vpc_peering_connection_id = aws_vpc_peering_connection.this.id
}

# Watchtower private subnets -> Conduct CIDR. (These aws_route resources are
# managed by Conduct's state but live in Watchtower's route tables — the
# "consumer owns the link" choice. They don't conflict with Watchtower's VPC
# module, which never defines a route for Conduct's CIDR.)
resource "aws_route" "accepter_to_requester" {
  count                     = length(local.accepter_route_table_ids)
  route_table_id            = local.accepter_route_table_ids[count.index]
  destination_cidr_block    = var.requester_cidr
  vpc_peering_connection_id = aws_vpc_peering_connection.this.id
}

output "peering_connection_id" {
  value = aws_vpc_peering_connection.this.id
}
