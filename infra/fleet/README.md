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
   changes. When the *policy documents* in the script change, paste
   the whole script again: it is idempotent and republishes the two
   deployer policies (`cubie-fleet-deployer`,
   `cubie-fleet-deployer-scoped`) as new default versions.
4. **Variables.** Copy `terraform.tfvars.example` to
   `terraform.tfvars` (gitignored) and fill in the App ID, key path,
   RunsOn license key (one license covers Flex and Fleet), and alert
   email. Networking needs no input: the stack creates its own VPC
   with public subnets in all three AZs (GPU spot pools span 2a/2b/2c
   and g5 exists only in 2a/2c; public-only means no NAT cost).

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

This stack owns its VPC and shares nothing with the Flex stack, so
once nothing else uses Flex its CloudFormation stack can be deleted
outright (CloudShell: `aws cloudformation delete-stack --stack-name
<flex-stack> --region ap-southeast-2`). If the delete ends in
DELETE_FAILED on the Flex S3 cache bucket, empty that bucket and
retry. The Flex GitHub App under the former owner's personal settings
can be uninstalled afterwards.

## Caching

Runners deliberately do **not** enable RunsOn's `s3-cache` (Magic
Cache) extra. It requires a `runs-on/action@v2` step in every job
(without it, the sidecar intercepts the GitHub artifact service and
every `actions/upload-artifact` call fails on a non-JSON
CreateArtifact response — observed live), and RunsOn documents that
the shared S3 cache bucket must not be enabled for runners that
public repositories can use — cubie is public. Workflow-level caching
(setup-uv) uses GitHub's cache service instead.

## Cost & timeline report

`cost_report.py` renders a single self-contained HTML report for one
`ci_cuda_tests.yml` run: per-leg boot / CI-step / shutdown bands, per-leg
cost at the achieved spot price, per-instance-type minutes and cost, the
run aggregate, a separate spot-capacity **wait** chart (wait is not
billed, so it is kept out of the runtime and cost charts), and account
24h-hourly / 30d-daily usage-by-instance-type and cost-by-service panels.

```powershell
python infra/fleet/cost_report.py <run-id> --out report.html
```

It correlates three data planes, keyed on the EC2 instance id RunsOn
embeds in each runner name (`runs-on--i-<id>--...`): the GitHub Actions
Jobs API (step timings), each leg's `Set up job` log (RunsOn boot
timeline, instance type/AZ, launch time), and AWS via the `cubie-fleet`
profile — `ec2:DescribeSpotPriceHistory` (achieved spot rate),
`cloudtrail:LookupEvents` (instance terminate time), and Cost Explorer
(`ce:GetCostAndUsage`) for the aggregate panels. The last two are the
read-only grants the bootstrap policy's `ReadOnly` /
`CostExplorerReadOnly` statements add; the CloudTrail and Cost Explorer
panels degrade to a note if they are absent.

Requirements: `gh` authenticated to the repo, the `cubie-fleet` AWS
profile, and `matplotlib`/`numpy`. Cost Explorer hourly granularity must
be enabled on the account for the 24h panel. Runs on Windows (it forces
UTF-8 on the AWS CLI subprocess, which otherwise dies rendering the
non-breaking spaces CloudTrail events carry). `--cache-dir` sets where
per-job logs are cached between runs.
