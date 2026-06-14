# _envcommon/alb.hcl — public Application Load Balancer fronting the api.
# HTTP(:80) redirects to HTTPS(:443); HTTPS forwards to the api target group.
# ECS registers task IPs into that target group (target_type = "ip", the
# awsvpc networking mode Fargate uses). The worker has no LB.
#
# If you have NO domain yet, see README "No domain yet?": drop the acm
# dependency + https listener and forward :80 straight to the target group.

locals {
  env = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
}

terraform {
  source = "tfr:///terraform-aws-modules/alb/aws//.?version=9.11.0"
}

dependency "vpc" {
  config_path = "../vpc"
  mock_outputs = {
    vpc_id         = "vpc-00000000000000000"
    public_subnets = ["subnet-pub-a", "subnet-pub-b", "subnet-pub-c"]
  }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}

dependency "sg" {
  config_path                             = "../security-groups"
  mock_outputs                            = { alb_sg_id = "sg-alb0000" }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}

dependency "acm" {
  config_path                             = "../acm"
  mock_outputs                            = { acm_certificate_arn = "arn:aws:acm:us-west-2:111122223333:certificate/mock" }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}

inputs = {
  name    = "conduct-${local.env.environment}"
  vpc_id  = dependency.vpc.outputs.vpc_id
  subnets = dependency.vpc.outputs.public_subnets

  create_security_group = false
  security_groups       = [dependency.sg.outputs.alb_sg_id]

  enable_deletion_protection = local.env.environment == "prod"

  target_groups = {
    api = {
      name_prefix = "cdt-"
      protocol    = "HTTP"
      port        = 8000
      target_type = "ip"
      health_check = {
        path                = "/health"
        matcher             = "200"
        healthy_threshold   = 2
        unhealthy_threshold = 3
        interval            = 15
        timeout             = 5
      }
      # ECS service attaches the targets, not the ALB module.
      create_attachment = false
    }
  }

  listeners = {
    http_redirect = {
      port     = 80
      protocol = "HTTP"
      redirect = { port = "443", protocol = "HTTPS", status_code = "HTTP_301" }
    }
    https = {
      port            = 443
      protocol        = "HTTPS"
      certificate_arn = dependency.acm.outputs.acm_certificate_arn
      forward         = { target_group_key = "api" }
    }
  }
}
