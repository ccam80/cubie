variable "aws_region" {
  description = "AWS region the fleet stack deploys into."
  type        = string
  default     = "ap-southeast-2"
}

variable "aws_profile" {
  description = "AWS CLI profile holding the short-term deployer credentials (see bootstrap/cloudshell-iam.sh)."
  type        = string
  default     = "cubie-fleet"
}

variable "stack_name" {
  description = "Name of the RunsOn Fleet stack; prefixes every AWS resource and the GitHub scale-set names."
  type        = string
  default     = "cubie-fleet"
}

variable "github_app_id" {
  description = "App ID of the RunsOn Fleet GitHub App installed on the organization that owns cubie."
  type        = number
}

variable "github_app_private_key_path" {
  description = "Path to the .pem private key generated for the RunsOn Fleet GitHub App."
  type        = string
}

variable "license_key" {
  description = "RunsOn license key (the same key as the existing Flex CloudFormation install; one license covers both products)."
  type        = string
  sensitive   = true
}

variable "alert_email" {
  description = "Email address RunsOn sends alerts to (requires confirmation on first deploy)."
  type        = string
}

variable "vpc_id" {
  description = "VPC to run Fargate and the GPU runners in. Reuse the existing RunsOn Flex stack's VPC to avoid a second VPC's cost."
  type        = string
}

variable "public_subnet_ids" {
  description = "Public subnet IDs in vpc_id used for the Fargate worker and runner instances."
  type        = list(string)
}

variable "public_route_table_id" {
  description = "Route table with the VPC's internet-gateway route; the extra public subnet associates with it."
  type        = string
}

variable "extra_public_subnet_cidr" {
  description = "CIDR for the additional public subnet this stack creates to widen GPU spot pools."
  type        = string
  default     = "10.1.32.0/20"
}

variable "extra_public_subnet_az" {
  description = "Availability zone for the additional public subnet."
  type        = string
  default     = "ap-southeast-2c"
}
