# _envcommon/ecs-service-worker.hcl — the RQ worker service.
# Same image + env + secrets + EFS + Bedrock role as the api, but runs the
# worker command, has NO load balancer, and isn't reachable inbound (its :8001
# Prometheus metrics port is scraped in-VPC, not via an LB).
#
# As with the api: locals is dependency-free; container env/secrets are built
# inline in `inputs`.

locals {
  account = read_terragrunt_config(find_in_parent_folders("account.hcl")).locals
  env     = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
  region  = read_terragrunt_config(find_in_parent_folders("region.hcl")).locals
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
dependency "rds" {
  config_path                             = "../rds"
  mock_outputs                            = { db_instance_identifier = "conduct-mock" }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}

inputs = {
  name          = "conduct-${local.env.environment}-worker"
  cluster_arn   = dependency.cluster.outputs.cluster_arn
  cpu           = local.env.worker_cpu
  memory        = local.env.worker_memory
  desired_count = local.env.worker_desired_count

  enable_execute_command = true

  runtime_platform = {
    cpu_architecture        = "ARM64"
    operating_system_family = "LINUX"
  }

  container_definitions = {
    conduct-worker = {
      essential     = true
      image         = "${dependency.ecr.outputs.repository_url}:${local.env.image_tag}"
      command       = ["python", "-m", "worker.queue"]
      port_mappings = [{ containerPort = 8001, protocol = "tcp" }] # Prometheus metrics

      environment = concat(
        [
          { name = "REDIS_URL", value = "redis://${dependency.redis.outputs.primary_endpoint}:6379/0" },
          { name = "OTEL_SERVICE_NAME", value = "conduct-worker" },
          { name = "DEFAULT_MODEL", value = local.env.default_model },
          { name = "DEFAULT_SENSITIVE_MODEL", value = local.env.default_sensitive_model },
          { name = "TTS_VOICES_DIR", value = "/app/voices" },
          { name = "TTS_OUTPUT_DIR", value = "/app/output" },
          { name = "AWS_REGION", value = local.region.aws_region },
        ],
        local.env.otlp_endpoint != "" ? [{ name = "OTEL_EXPORTER_OTLP_ENDPOINT", value = local.env.otlp_endpoint }] : [],
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

  # No load_balancer block — the worker takes no inbound traffic.

  task_exec_secret_arns = [for k, v in dependency.secrets.outputs.secret_arns : v]

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
