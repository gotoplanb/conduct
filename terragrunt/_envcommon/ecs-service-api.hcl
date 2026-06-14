# _envcommon/ecs-service-api.hcl — the public-facing FastAPI service.
# Fargate task running the conduct image with the uvicorn command, behind the
# ALB, mounting the shared EFS at /app/output, reading secrets from Secrets
# Manager, and granted Bedrock invocation on its task role.
#
# Design note: `locals` holds ONLY hierarchy-derived values (no dependency
# references) — Terragrunt allows a single locals block, and keeping it free of
# dependency.* keeps it resolvable at `hcl validate` time. The container's
# environment + secrets (which DO reference dependency outputs) are built inline
# in `inputs`, which is the right place for dependency wiring.

locals {
  account = read_terragrunt_config(find_in_parent_folders("account.hcl")).locals
  env     = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
  region  = read_terragrunt_config(find_in_parent_folders("region.hcl")).locals

  fqdn = local.env.environment == "prod" ? local.account.domain_base : "${local.env.environment}.${local.account.domain_base}"
  public_url = (
    local.env.public_url_override != "" ? local.env.public_url_override :
    (local.account.domain_base != "" ? "https://${local.fqdn}" : "")
  )
}

terraform {
  source = "tfr:///terraform-aws-modules/ecs/aws//modules/service?version=5.12.0"
}

dependency "cluster" {
  config_path                             = "../ecs-cluster"
  mock_outputs                            = { cluster_arn = "arn:aws:ecs:us-west-2:111122223333:cluster/mock" }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}
dependency "vpc" {
  config_path                             = "../vpc"
  mock_outputs                            = { private_subnets = ["subnet-a", "subnet-b", "subnet-c"] }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}
dependency "sg" {
  config_path                             = "../security-groups"
  mock_outputs                            = { app_sg_id = "sg-app0000" }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}
dependency "alb" {
  config_path                             = "../alb"
  mock_outputs                            = { target_groups = { api = { arn = "arn:aws:elasticloadbalancing:us-west-2:111122223333:targetgroup/mock/0" } } }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}
dependency "ecr" {
  config_path                             = "../ecr"
  mock_outputs                            = { repository_url = "111122223333.dkr.ecr.us-west-2.amazonaws.com/conduct-mock" }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}
dependency "redis" {
  config_path                             = "../redis"
  mock_outputs                            = { primary_endpoint = "mock.cache.amazonaws.com" }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}
dependency "efs" {
  config_path = "../efs"
  mock_outputs = {
    id            = "fs-00000000"
    arn           = "arn:aws:elasticfilesystem:us-west-2:111122223333:file-system/fs-00000000"
    access_points = { output = { id = "fsap-00000000" } }
  }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}
dependency "secrets" {
  config_path = "../secrets"
  mock_outputs = {
    secret_arns = {
      "database-url"      = "arn:aws:secretsmanager:us-west-2:111122223333:secret:conduct/x/database-url"
      "admin-key"         = "arn:aws:secretsmanager:us-west-2:111122223333:secret:conduct/x/admin-key"
      "secrets-key"       = "arn:aws:secretsmanager:us-west-2:111122223333:secret:conduct/x/secrets-key"
      "anthropic-api-key" = "arn:aws:secretsmanager:us-west-2:111122223333:secret:conduct/x/anthropic-api-key"
    }
  }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}
# Ordering only: ensure the DB exists before tasks start (avoids boot crashloop).
dependency "rds" {
  config_path                             = "../rds"
  mock_outputs                            = { db_instance_identifier = "conduct-mock" }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}

inputs = {
  name          = "conduct-${local.env.environment}-api"
  cluster_arn   = dependency.cluster.outputs.cluster_arn
  cpu           = local.env.api_cpu
  memory        = local.env.api_memory
  desired_count = local.env.api_desired_count

  enable_execute_command = true # `aws ecs execute-command` shell-in for debugging

  # Graviton (cheaper). Requires arm64 images: build with
  # `docker buildx build --platform linux/arm64`. Switch to X86_64 if you push amd64.
  runtime_platform = {
    cpu_architecture        = "ARM64"
    operating_system_family = "LINUX"
  }

  container_definitions = {
    conduct-api = {
      essential     = true
      image         = "${dependency.ecr.outputs.repository_url}:${local.env.image_tag}"
      command       = ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
      port_mappings = [{ containerPort = 8000, protocol = "tcp" }]

      environment = concat(
        [
          { name = "REDIS_URL", value = "redis://${dependency.redis.outputs.primary_endpoint}:6379/0" },
          { name = "OTEL_SERVICE_NAME", value = "conduct" },
          { name = "DEFAULT_MODEL", value = local.env.default_model },
          { name = "DEFAULT_SENSITIVE_MODEL", value = local.env.default_sensitive_model },
          { name = "TTS_VOICES_DIR", value = "/app/voices" },
          { name = "TTS_OUTPUT_DIR", value = "/app/output" },
          # Bedrock uses the default credential chain (the task role) + region.
          { name = "AWS_REGION", value = local.region.aws_region },
        ],
        local.public_url != "" ? [{ name = "CONDUCT_PUBLIC_URL", value = local.public_url }] : [],
        local.env.otlp_endpoint != "" ? [{ name = "OTEL_EXPORTER_OTLP_ENDPOINT", value = local.env.otlp_endpoint }] : [],
        local.env.grafana_url != "" ? [{ name = "GRAFANA_BASE_URL", value = local.env.grafana_url }] : [],
      )

      secrets = [
        { name = "DATABASE_URL", valueFrom = dependency.secrets.outputs.secret_arns["database-url"] },
        { name = "CONDUCT_ADMIN_KEY", valueFrom = dependency.secrets.outputs.secret_arns["admin-key"] },
        { name = "CONDUCT_SECRETS_KEY", valueFrom = dependency.secrets.outputs.secret_arns["secrets-key"] },
        { name = "ANTHROPIC_API_KEY", valueFrom = dependency.secrets.outputs.secret_arns["anthropic-api-key"] },
      ]

      mount_points             = [{ sourceVolume = "output", containerPath = "/app/output", readOnly = false }]
      readonly_root_filesystem = false
    }
  }

  volume = {
    output = {
      efs_volume_configuration = {
        file_system_id     = dependency.efs.outputs.id
        transit_encryption = "ENABLED"
        authorization_config = {
          access_point_id = dependency.efs.outputs.access_points["output"].id
          iam             = "ENABLED"
        }
      }
    }
  }

  subnet_ids            = dependency.vpc.outputs.private_subnets
  security_group_ids    = [dependency.sg.outputs.app_sg_id]
  create_security_group = false
  assign_public_ip      = false

  load_balancer = {
    api = {
      target_group_arn = dependency.alb.outputs.target_groups["api"].arn
      container_name   = "conduct-api"
      container_port   = 8000
    }
  }

  # Execution role may read the app secrets (to inject them at task start).
  task_exec_secret_arns = [for k, v in dependency.secrets.outputs.secret_arns : v]

  # Task role: invoke Bedrock + use the EFS access point.
  tasks_iam_role_statements = [
    {
      effect    = "Allow"
      actions   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
      resources = ["*"]
    },
    {
      effect    = "Allow"
      actions   = ["elasticfilesystem:ClientMount", "elasticfilesystem:ClientWrite"]
      resources = [dependency.efs.outputs.arn]
    },
  ]
}
