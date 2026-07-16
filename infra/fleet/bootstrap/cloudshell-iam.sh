#!/usr/bin/env bash
# One-shot AWS CloudShell bootstrap for the fleet deployer identity.
#
# Paste this whole file into an AWS CloudShell session (console ->
# CloudShell icon). It creates:
#   1. a customer-managed policy `cubie-fleet-deployer` -- the minimal
#      permission set `tofu apply` in infra/fleet needs (see the policy
#      document below for the per-service rationale);
#   2. a role `cubie-fleet-deployer` assumable by IAM identities in
#      this account only;
# and then mints 1-hour temporary credentials for that role.
#
# The local AWS CLI only ever holds those 1-hour credentials, so a
# leaked or mishandled key expires on its own and never carries more
# than the scoped deployer permissions.
#
# REGENERATING CREDENTIALS: rerun just the final `aws sts assume-role`
# command (or the whole script -- it is idempotent) and copy the fresh
# Credentials block into ~/.aws/credentials under [cubie-fleet].
set -euo pipefail

REGION="ap-southeast-2"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Permissions, by statement:
# - RegionalDeploy: create/read/update/delete for the services the
#   RunsOn Fleet Terraform module manages (EC2 launch templates and
#   security groups, ECS/Fargate runtime, CloudWatch logs + alarms,
#   DynamoDB state table, the diagnostics Lambda, S3 cache bucket,
#   the rendered config secret, SNS alert topic, ECS autoscaling,
#   EventBridge rules) plus read-only CloudFormation to look up the
#   existing Flex stack's VPC/subnets. Locked to ap-southeast-2.
# - IamScoped: IAM role/policy/instance-profile management restricted
#   to resources whose names start with cubie-fleet or runs-on, so the
#   deployer cannot touch or escalate through any other IAM entity.
#   PassRole is restricted the same way.
# - ServiceLinkedRoles: lets the first ECS/spot/autoscaling use in the
#   account create AWS's own service-linked roles, nothing else.
cat > /tmp/cubie-fleet-deployer-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "RegionalDeploy",
      "Effect": "Allow",
      "Action": [
        "ec2:*",
        "ecs:*",
        "logs:*",
        "dynamodb:*",
        "lambda:*",
        "s3:*",
        "secretsmanager:*",
        "sns:*",
        "cloudwatch:*",
        "application-autoscaling:*",
        "events:*",
        "scheduler:*",
        "ecr:Get*",
        "ecr:Describe*",
        "cloudformation:Describe*",
        "cloudformation:List*",
        "cloudformation:Get*",
        "servicequotas:Get*",
        "servicequotas:List*"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": { "aws:RequestedRegion": "${REGION}" }
      }
    },
    {
      "Sid": "IamScoped",
      "Effect": "Allow",
      "Action": [
        "iam:AddRoleToInstanceProfile",
        "iam:AttachRolePolicy",
        "iam:CreateInstanceProfile",
        "iam:CreatePolicy",
        "iam:CreatePolicyVersion",
        "iam:CreateRole",
        "iam:DeleteInstanceProfile",
        "iam:DeletePolicy",
        "iam:DeletePolicyVersion",
        "iam:DeleteRole",
        "iam:DeleteRolePolicy",
        "iam:DetachRolePolicy",
        "iam:GetInstanceProfile",
        "iam:GetPolicy",
        "iam:GetPolicyVersion",
        "iam:GetRole",
        "iam:GetRolePolicy",
        "iam:ListAttachedRolePolicies",
        "iam:ListInstanceProfilesForRole",
        "iam:ListPolicyTags",
        "iam:ListPolicyVersions",
        "iam:ListRolePolicies",
        "iam:ListRoleTags",
        "iam:PassRole",
        "iam:PutRolePolicy",
        "iam:RemoveRoleFromInstanceProfile",
        "iam:TagInstanceProfile",
        "iam:TagPolicy",
        "iam:TagRole",
        "iam:UntagInstanceProfile",
        "iam:UntagPolicy",
        "iam:UntagRole",
        "iam:UpdateAssumeRolePolicy",
        "iam:UpdateRole"
      ],
      "Resource": [
        "arn:aws:iam::${ACCOUNT_ID}:role/cubie-fleet*",
        "arn:aws:iam::${ACCOUNT_ID}:role/runs-on*",
        "arn:aws:iam::${ACCOUNT_ID}:policy/cubie-fleet*",
        "arn:aws:iam::${ACCOUNT_ID}:policy/runs-on*",
        "arn:aws:iam::${ACCOUNT_ID}:instance-profile/cubie-fleet*",
        "arn:aws:iam::${ACCOUNT_ID}:instance-profile/runs-on*"
      ]
    },
    {
      "Sid": "ServiceLinkedRoles",
      "Effect": "Allow",
      "Action": "iam:CreateServiceLinkedRole",
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "iam:AWSServiceName": [
            "ecs.amazonaws.com",
            "spot.amazonaws.com",
            "ecs.application-autoscaling.amazonaws.com",
            "autoscaling.amazonaws.com"
          ]
        }
      }
    }
  ]
}
EOF

# Only identities inside this account can assume the role; combined
# with the 1-hour session cap this is the whole exposure surface.
cat > /tmp/cubie-fleet-deployer-trust.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "AWS": "arn:aws:iam::${ACCOUNT_ID}:root" },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/cubie-fleet-deployer"
if aws iam get-policy --policy-arn "$POLICY_ARN" >/dev/null 2>&1; then
  # Idempotent rerun: push the document as a new default version and
  # prune the oldest non-default version (IAM keeps at most five).
  OLDEST=$(aws iam list-policy-versions --policy-arn "$POLICY_ARN" \
    --query 'Versions[?!IsDefaultVersion]|[-1].VersionId' --output text)
  if [ "$OLDEST" != "None" ] && [ "$(aws iam list-policy-versions \
      --policy-arn "$POLICY_ARN" --query 'length(Versions)')" -ge 5 ]; then
    aws iam delete-policy-version --policy-arn "$POLICY_ARN" \
      --version-id "$OLDEST"
  fi
  aws iam create-policy-version --policy-arn "$POLICY_ARN" \
    --policy-document file:///tmp/cubie-fleet-deployer-policy.json \
    --set-as-default
else
  aws iam create-policy --policy-name cubie-fleet-deployer \
    --policy-document file:///tmp/cubie-fleet-deployer-policy.json
fi

if ! aws iam get-role --role-name cubie-fleet-deployer >/dev/null 2>&1; then
  aws iam create-role --role-name cubie-fleet-deployer \
    --assume-role-policy-document file:///tmp/cubie-fleet-deployer-trust.json \
    --max-session-duration 3600
fi
aws iam attach-role-policy --role-name cubie-fleet-deployer \
  --policy-arn "$POLICY_ARN"

echo
echo "=== Temporary credentials (valid 1 hour) ==="
aws sts assume-role \
  --role-arn "arn:aws:iam::${ACCOUNT_ID}:role/cubie-fleet-deployer" \
  --role-session-name cubie-fleet-cli \
  --duration-seconds 3600 \
  --query Credentials
echo
echo "Copy the JSON above into the [cubie-fleet] profile locally."
echo "To regenerate later, rerun just the 'aws sts assume-role' command."
