# RunsOn Fleet stack (GPU CI runners)

OpenTofu configuration for the RunsOn **Fleet** stack that provides
the GPU runners for `.github/workflows/ci_cuda_tests.yml`.

## Why Fleet

The account's "All G and VT Spot" quota is a fixed 8 vCPU and the
On-Demand G/VT quota is 0 (not grantable in ap-southeast-2). Under the
old Flex setup, each queued matrix job's webhook launched an instance
immediately — Flex has no `max-parallel` awareness — so the matrix
overran the quota, quota rejections tripped Flex's fixed 5-minute
on-demand snooze, and with 0 on-demand quota those legs simply failed.
The workflow worked around this with 4-vCPU runners plus a
re-dispatching `retry` job.

Fleet uses GitHub runner **scale sets**: jobs queue on the GitHub side
and the Fleet runtime launches EC2 capacity per *assigned* job, so
`strategy.max-parallel` genuinely bounds instance demand and the retry
apparatus goes away. The workflow runs `max-parallel: 2`: two xlarge
(4-vCPU) legs fit the quota concurrently, and a leg lands a 2xlarge
when ap-southeast-2 has that capacity (rare — spot placement score
1/10 for 2xlarge vs 9/10 for xlarge). On a launch failure the Fleet
runtime retries with backoff while the job stays queued, so capacity
droughts cost latency, not red legs.

The fleets have no `schedule` (warm standby) on purpose: RunsOn warm
pools use on-demand capacity, which this account cannot launch for G
instances. All capacity is cold spot launches.

## One-time setup

1. **GitHub organization.** Fleet registers organization-scoped runner
   scale sets; personal accounts are not supported. The repo must live
   in an organization (a free plan works — scale sets register into
   the org's default runner group).
2. **GitHub App** (organization mode). Create it from the pre-filled
   link (replace `<ORG>`):

   ```text
   https://github.com/organizations/<ORG>/settings/apps/new?name=RunsOn%20Fleet%20%5B<ORG>%5D&url=https%3A%2F%2Fruns-on.com&public=false&webhook_active=false&organization_self_hosted_runners=write&actions=read
   ```

   Generate a private key (.pem), install the App on the organization,
   and note the App ID.
3. **AWS deployer credentials.** Paste
   [`bootstrap/cloudshell-iam.sh`](bootstrap/cloudshell-iam.sh) into an
   AWS CloudShell session. It creates a name-scoped, region-locked
   deployer role and prints 1-hour temporary credentials. Put them in
   `~/.aws/credentials`:

   ```ini
   [cubie-fleet]
   aws_access_key_id     = ...
   aws_secret_access_key = ...
   aws_session_token     = ...
   ```

   When they expire, rerun the script's final `aws sts assume-role`
   command in CloudShell and paste the fresh block — nothing else
   changes.
4. **Variables.** Copy `terraform.tfvars.example` to
   `terraform.tfvars` (gitignored) and fill in the App ID, key path,
   RunsOn license key (the Flex install's key; one license covers both
   products), alert email, and the Flex stack's VPC/subnet IDs.

## Deploy

```powershell
cd infra/fleet
tofu init
tofu plan
tofu apply
```

State is local (`terraform.tfstate`, gitignored) and contains the
license key and App private key — keep it on the machine that manages
the stack.

After apply, the scale sets `cubie-fleet-gpu-linux` and
`cubie-fleet-gpu-windows` appear under the organization's Actions
runner settings, and workflows target them with:

```yaml
runs-on: runs-on/fleet=gpu-linux/env=production
```

## Relationship to the Flex install

The existing Flex CloudFormation stack is untouched by this stack,
but its GitHub App must **not** have access to this repository. Flex
(v3.1.3) claims any workflow job whose label starts with `runs-on/`,
so with both apps attached every fleet-labeled job also triggers Flex,
which parses no `runner=` key, falls back to its default 2-vCPU
runner, and launches duplicate instances for jobs it can never run
(observed: 3 launch attempts per job, spot then on-demand). Repo-
scoped app installations survive repository transfers, so after
moving the repo into an organization, detach it explicitly: the app
owner's personal settings -> Applications -> the Flex RunsOn app ->
Configure -> remove this repository.

Once Fleet has proven itself on the CUDA matrix, the Flex stack can
be deleted from CloudFormation if nothing else uses it.
