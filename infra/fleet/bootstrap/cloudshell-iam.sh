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
# - ReadOnly: region-locked Describe/Get/List across the services the
#   RunsOn Fleet Terraform module touches, plus read-only
#   CloudFormation and Service Quotas for diagnostics. Reads carry no
#   secret material: secretsmanager:GetSecretValue is NOT here -- it
#   lives in SecretsScoped, bound to this stack's secret prefix.
# - Ec2Provision: the non-destructive EC2 mutations an apply needs
#   (VPC/subnet/route/SG/launch-template creation and tagging).
#   RunInstances is deliberately absent: the deployer never launches
#   instances -- only the fleet runtime's own role does.
# - Ec2ScopedDestroy: terminate/delete only for EC2 resources tagged
#   stack=cubie-fleet, so the credentials cannot touch instances,
#   VPCs, or security groups belonging to anything else.
# - S3/Secrets/Dynamo/Lambda/Logs/Sns/Ecs/Events/Cw *Scoped: full
#   management bound to this stack's ARN name prefixes. Resource ARNs
#   bind regardless of which endpoint a call reaches, which also
#   covers S3 calls made through the global endpoint.
# - IamScoped: IAM role/policy/instance-profile management restricted
#   to names starting with cubie-fleet or runs-on, PassRole included,
#   so the deployer cannot escalate through any other IAM entity.
# - ServiceLinkedRoles: lets the first ECS/spot/autoscaling use in the
#   account create AWS's own service-linked roles, nothing else.
cat > /tmp/cubie-fleet-deployer-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadOnly",
      "Effect": "Allow",
      "Action": [
        "ec2:Describe*",
        "ec2:Get*",
        "ecs:Describe*",
        "ecs:List*",
        "logs:Describe*",
        "logs:Get*",
        "logs:List*",
        "logs:FilterLogEvents",
        "logs:StartQuery",
        "logs:StopQuery",
        "logs:PutQueryDefinition",
        "logs:DeleteQueryDefinition",
        "cloudformation:Describe*",
        "cloudformation:List*",
        "cloudformation:Get*",
        "servicequotas:Get*",
        "servicequotas:List*",
        "sns:Get*",
        "sns:List*",
        "cloudwatch:Describe*",
        "cloudwatch:Get*",
        "cloudwatch:List*",
        "application-autoscaling:*",
        "events:Describe*",
        "events:List*",
        "scheduler:Get*",
        "scheduler:List*",
        "dynamodb:Describe*",
        "dynamodb:List*",
        "lambda:Get*",
        "lambda:List*",
        "secretsmanager:Describe*",
        "secretsmanager:List*",
        "ssm:DescribeParameters",
        "ecr:Get*",
        "ecr:Describe*"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": { "aws:RequestedRegion": "${REGION}" }
      }
    },
    {
      "Sid": "Ec2Provision",
      "Effect": "Allow",
      "Action": [
        "ec2:CreateVpc",
        "ec2:ModifyVpcAttribute",
        "ec2:CreateSubnet",
        "ec2:ModifySubnetAttribute",
        "ec2:CreateInternetGateway",
        "ec2:AttachInternetGateway",
        "ec2:CreateRouteTable",
        "ec2:CreateRoute",
        "ec2:ReplaceRoute",
        "ec2:DeleteRoute",
        "ec2:AssociateRouteTable",
        "ec2:DisassociateRouteTable",
        "ec2:CreateSecurityGroup",
        "ec2:AuthorizeSecurityGroupIngress",
        "ec2:AuthorizeSecurityGroupEgress",
        "ec2:RevokeSecurityGroupIngress",
        "ec2:RevokeSecurityGroupEgress",
        "ec2:ModifySecurityGroupRules",
        "ec2:CreateLaunchTemplate",
        "ec2:CreateLaunchTemplateVersion",
        "ec2:ModifyLaunchTemplate",
        "ec2:DeleteLaunchTemplateVersions",
        "ec2:CreateTags",
        "ec2:DeleteTags"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": { "aws:RequestedRegion": "${REGION}" }
      }
    },
    {
      "Sid": "Ec2ScopedDestroy",
      "Effect": "Allow",
      "Action": [
        "ec2:TerminateInstances",
        "ec2:DeleteVpc",
        "ec2:DeleteSubnet",
        "ec2:DeleteInternetGateway",
        "ec2:DetachInternetGateway",
        "ec2:DeleteRouteTable",
        "ec2:DeleteSecurityGroup",
        "ec2:DeleteLaunchTemplate"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": "${REGION}",
          "aws:ResourceTag/stack": "cubie-fleet"
        }
      }
    },
    {
      "Sid": "S3Scoped",
      "Effect": "Allow",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::cubie-fleet-*",
        "arn:aws:s3:::cubie-fleet-*/*"
      ]
    },
    {
      "Sid": "SecretsScoped",
      "Effect": "Allow",
      "Action": "secretsmanager:*",
      "Resource": [
        "arn:aws:secretsmanager:${REGION}:${ACCOUNT_ID}:secret:/runs-on/cubie-fleet/*",
        "arn:aws:secretsmanager:${REGION}:${ACCOUNT_ID}:secret:cubie-fleet*"
      ]
    },
    {
      "Sid": "DynamoScoped",
      "Effect": "Allow",
      "Action": "dynamodb:*",
      "Resource": [
        "arn:aws:dynamodb:${REGION}:${ACCOUNT_ID}:table/cubie-fleet-*",
        "arn:aws:dynamodb:${REGION}:${ACCOUNT_ID}:table/cubie-fleet-*/index/*"
      ]
    },
    {
      "Sid": "LambdaScoped",
      "Effect": "Allow",
      "Action": "lambda:*",
      "Resource": "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:cubie-fleet-*"
    },
    {
      "Sid": "LogsScoped",
      "Effect": "Allow",
      "Action": "logs:*",
      "Resource": [
        "arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:*cubie-fleet*",
        "arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:*cubie-fleet*:*"
      ]
    },
    {
      "Sid": "SnsScoped",
      "Effect": "Allow",
      "Action": "sns:*",
      "Resource": "arn:aws:sns:${REGION}:${ACCOUNT_ID}:cubie-fleet-*"
    },
    {
      "Sid": "CloudWatchScoped",
      "Effect": "Allow",
      "Action": "cloudwatch:*",
      "Resource": [
        "arn:aws:cloudwatch:${REGION}:${ACCOUNT_ID}:alarm:cubie-fleet*",
        "arn:aws:cloudwatch::${ACCOUNT_ID}:dashboard/cubie-fleet*"
      ]
    },
    {
      "Sid": "EcsScoped",
      "Effect": "Allow",
      "Action": "ecs:*",
      "Resource": [
        "arn:aws:ecs:${REGION}:${ACCOUNT_ID}:cluster/cubie-fleet*",
        "arn:aws:ecs:${REGION}:${ACCOUNT_ID}:service/cubie-fleet*/*",
        "arn:aws:ecs:${REGION}:${ACCOUNT_ID}:task/cubie-fleet*/*",
        "arn:aws:ecs:${REGION}:${ACCOUNT_ID}:task-definition/cubie-fleet-*:*",
        "arn:aws:ecs:${REGION}:${ACCOUNT_ID}:container-instance/cubie-fleet*/*"
      ]
    },
    {
      "Sid": "EcsUnscoped",
      "Effect": "Allow",
      "Action": [
        "ecs:CreateCluster",
        "ecs:RegisterTaskDefinition",
        "ecs:DeregisterTaskDefinition"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": { "aws:RequestedRegion": "${REGION}" }
      }
    },
    {
      "Sid": "EventsScoped",
      "Effect": "Allow",
      "Action": [
        "events:*"
      ],
      "Resource": "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/cubie-fleet*"
    },
    {
      "Sid": "SchedulerScoped",
      "Effect": "Allow",
      "Action": "scheduler:*",
      "Resource": [
        "arn:aws:scheduler:${REGION}:${ACCOUNT_ID}:schedule/*/cubie-fleet*",
        "arn:aws:scheduler:${REGION}:${ACCOUNT_ID}:schedule-group/cubie-fleet*"
      ]
    },
    {
      "Sid": "SsmScoped",
      "Effect": "Allow",
      "Action": "ssm:*",
      "Resource": "arn:aws:ssm:${REGION}:${ACCOUNT_ID}:parameter/cubie-fleet*"
    },
    {
      "Sid": "ResourceGroupsScoped",
      "Effect": "Allow",
      "Action": "resource-groups:*",
      "Resource": "arn:aws:resource-groups:${REGION}:${ACCOUNT_ID}:group/cubie-fleet*"
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
