# _envcommon/ecr.hcl — the container image repository for the conduct image.
#
# ECR is REGIONAL: a repo in us-west-2 is not visible to ECS in us-east-2. So
# there is one ecr unit per region, and your build/push step must push the
# image to each region's repo (or configure ECR cross-region replication —
# see terragrunt/README.md). api and worker share one image, so one repo.

locals {
  region = read_terragrunt_config(find_in_parent_folders("region.hcl")).locals
  env    = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
}

terraform {
  source = "tfr:///terraform-aws-modules/ecr/aws//.?version=2.3.1"
}

inputs = {
  repository_name                 = "conduct-${local.env.environment}"
  repository_image_tag_mutability = "IMMUTABLE" # tags are git SHAs; never overwrite

  # Keep the last 20 images; expire older untagged layers.
  repository_lifecycle_policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep last 20 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 20 }
      action       = { type = "expire" }
    }]
  })

  repository_image_scan_on_push = true
}
