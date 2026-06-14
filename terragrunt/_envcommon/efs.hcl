# _envcommon/efs.hcl — shared filesystem for Conduct's /app/output directory.
#
# WHY EFS: in docker-compose the api and worker bind-mount the SAME ./output
# host dir — the worker writes generated media (TTS mp3s, etc.) and the api
# serves them from /output/. On Fargate, api and worker are SEPARATE tasks on
# separate hosts, so a local volume can't be shared. EFS is a POSIX filesystem
# both tasks mount over NFS — zero app changes. (The cleaner long-term answer
# is to push artifacts to S3 and serve via presigned URLs / CloudFront; noted
# in the README as a follow-up.)
#
# An access point pins the mount to /output with a fixed POSIX uid/gid (1001,
# the "conduct" user from the Dockerfile) so file ownership is correct.

locals {
  region = read_terragrunt_config(find_in_parent_folders("region.hcl")).locals
  env    = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
}

terraform {
  source = "tfr:///terraform-aws-modules/efs/aws//.?version=1.6.5"
}

dependency "vpc" {
  config_path                             = "../vpc"
  mock_outputs                            = { private_subnets = ["subnet-aaa", "subnet-bbb", "subnet-ccc"] }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}

dependency "sg" {
  config_path                             = "../security-groups"
  mock_outputs                            = { efs_sg_id = "sg-efs0000" }
  mock_outputs_allowed_terraform_commands = ["plan", "validate", "init"]
}

inputs = {
  name      = "conduct-${local.env.environment}-output"
  encrypted = true

  # One mount target per private subnet, locked to the efs SG.
  mount_targets = {
    for i, subnet in dependency.vpc.outputs.private_subnets :
    local.region.azs[i] => { subnet_id = subnet }
  }
  security_group_ids    = [dependency.sg.outputs.efs_sg_id]
  create_security_group = false

  access_points = {
    output = {
      posix_user = { uid = 1001, gid = 1001 }
      root_directory = {
        path          = "/output"
        creation_info = { owner_uid = 1001, owner_gid = 1001, permissions = "0755" }
      }
    }
  }
}
