# RunsOn Fleet stack for cubie's GPU CI (ci_cuda_tests.yml).
#
# Fleet (runs-on.com/docs/flex-vs-fleet/) registers GitHub runner scale
# sets and launches EC2 capacity from *assigned-job* demand, so GitHub's
# `strategy.max-parallel: 1` serialises the CUDA matrix on the RunsOn
# side as well: at most one runner instance exists at a time. This is
# what lets each matrix leg use a full 8-vCPU (2xlarge) GPU instance
# inside the fixed 8-vCPU "All G and VT Spot" quota (on-demand G/VT
# quota is 0 in ap-southeast-2 and not grantable).
#
# Deliberately no `schedule` (hot/stopped standby) on the fleets: warm
# pool inventory uses on-demand EC2 capacity, which this account cannot
# launch for G instances. All capacity comes from cold spot launches.

terraform {
  required_version = ">= 1.5.7"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 6.45"
    }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile
}

locals {
  # Custom image catalog. Same contract as the Flex .github/runs-on.yml
  # `images:` block: RunsOn picks the newest AMI matching `name` owned
  # by this account ("self"), so the Packer bake workflow
  # (.github/workflows/build-windows-gpu-ami.yml) keeps working
  # unchanged.
  images = {
    cubie-win-gpu = {
      platform = "windows"
      arch     = "x64"
      owner    = "self"
      name     = "cubie-win-gpu-*"
    }
  }

  # Runner shapes. Both xlarge (4 vCPU) and 2xlarge (8 vCPU) sizes of
  # three GPU families (T4/A10G/L4); the single AWS GRID driver package
  # in the baked AMI covers all of them, and the workflow's
  # `pytest -n logical` scales the worker count to whichever size a
  # leg lands on. 2xlarge is preferred but spot placement scores show
  # ap-southeast-2 is structurally starved of 2xlarge GPU capacity
  # (score 1/10 vs 9/10 for xlarge), so xlarge keeps the suite running;
  # price-capacity-optimized allocation will usually pick it. Either
  # size fits the fixed 8-vCPU G/VT spot quota, and max-parallel: 1 in
  # the workflow keeps demand to a single instance.
  runners = {
    gpu-linux-2xl = {
      family = [
        "g4dn.2xlarge", "g5.2xlarge", "g6.2xlarge",
        "g4dn.xlarge", "g5.xlarge", "g6.xlarge",
      ]
      image  = "ubuntu24-gpu-x64"
      spot   = "price-capacity-optimized"
      extras = ["s3-cache"]
    }
    gpu-windows-2xl = {
      family = [
        "g4dn.2xlarge", "g5.2xlarge", "g6.2xlarge",
        "g4dn.xlarge", "g5.xlarge", "g6.xlarge",
      ]
      image  = "cubie-win-gpu"
      spot   = "price-capacity-optimized"
      extras = ["s3-cache"]
    }
  }

  # One fleet per OS; each maps to one GitHub runner scale set named
  # <stack_name>-<fleet name>. Workflows target them with
  #   runs-on: runs-on/fleet=gpu-linux/env=production
  # No runner_group: scale sets register into the organization's
  # default runner group (custom groups need a paid GitHub plan).
  fleets = {
    gpu-linux = {
      timezone = "UTC"
      runner   = "gpu-linux-2xl"
    }
    gpu-windows = {
      timezone = "UTC"
      runner   = "gpu-windows-2xl"
    }
  }
}

# Third-AZ public subnet for the runners. The reused Flex VPC only has
# public subnets in ap-southeast-2a/2b, but g4dn/g5/g6 2xlarge are all
# offered in 2c (and 2c is the only other AZ with g5 at all), so this
# subnet widens the spot pools from 5 to 8 -- GPU spot capacity in this
# region is thin enough at 8 vCPU that the extra pools matter.
resource "aws_subnet" "public_extra" {
  vpc_id                  = var.vpc_id
  cidr_block              = var.extra_public_subnet_cidr
  availability_zone       = var.extra_public_subnet_az
  map_public_ip_on_launch = true

  tags = {
    Name    = "${var.stack_name}-public-${var.extra_public_subnet_az}"
    stack   = var.stack_name
    project = "cubie"
  }
}

resource "aws_route_table_association" "public_extra" {
  subnet_id      = aws_subnet.public_extra.id
  route_table_id = var.public_route_table_id
}

module "runs_on_fleet" {
  source  = "runs-on/runs-on/aws//modules/fleet"
  version = "3.1.3"

  stack_name  = var.stack_name
  environment = "production"

  # Organization mode: a GitHub App installed on exactly one
  # organization, with organization self-hosted runner write access.
  github_app_id          = var.github_app_id
  github_app_private_key = file(var.github_app_private_key_path)

  license_key = var.license_key
  email       = var.alert_email

  images  = local.images
  runners = local.runners
  fleets  = local.fleets

  vpc_id = var.vpc_id
  public_subnet_ids = concat(
    var.public_subnet_ids,
    [aws_subnet.public_extra.id],
  )

  # CI runs three times a week; fargate_spot keeps the always-on Fleet
  # worker's idle cost down, and the Fleet runtime reconciles any
  # in-flight jobs if the Fargate task is interrupted.
  app_size              = "small"
  app_capacity_provider = "fargate_spot"

  # A full-matrix leg (install + flake8 + real-GPU pytest) fits well
  # inside an hour today; 120 leaves headroom for suite growth without
  # letting a hung leg hold the quota for long.
  runner_max_runtime = 120

  tags = {
    project = "cubie"
  }
}
