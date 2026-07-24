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

## Cost & timeline dashboard

`cost_dashboard.py` serves a local interactive dashboard for GPU CI cost
and timing.

```powershell
python infra/fleet/cost_dashboard.py    # opens http://localhost:8787
```

Pick a run from the dropdown (recent `ci_cuda_tests.yml` runs, fetched on
demand) to see, per leg: a timeline of spot-capacity wait / boot / CI
steps / shutdown with the run total broken down beside it; time in each
CI step (with a run-total bar beside it); cost at the achieved spot
price; minutes and cost per instance type with the average spot rate
annotated; and spot-capacity wait per leg. The account section takes
inclusive from/to date pickers and a granularity and charts whole-account
usage hours per instance type and gross usage $ by service. Hourly ranges
are limited to 366 inclusive days and daily ranges to 3,660 days.

It correlates three data planes, keyed on the EC2 instance id RunsOn
embeds in each runner name (`runs-on--i-<id>--...`): the GitHub Actions
Jobs API (step timings), each leg's `Set up job` log (RunsOn boot
timeline, instance type/AZ, launch time), and AWS via the `cubie-fleet`
profile — `ec2:DescribeSpotPriceHistory` (achieved spot rate),
`cloudtrail:LookupEvents` (instance terminate time), and Cost Explorer
(`ce:GetCostAndUsage`) for the account panels. The last two are the
read-only grants the bootstrap policy's `ReadOnly` / `CostExplorerReadOnly`
statements add.

**Cost of use:** per-run views are free (GitHub API, `ec2:Describe*` and
`cloudtrail:LookupEvents` carry no charge). Only the account panels touch
Cost Explorer, billed $0.01 per `GetCostAndUsage` request. Account usage is
sourced from **hourly** Cost Explorer data and daily values are aggregated
from it (this matches CE's own daily totals to the cent).

The dashboard owns a transactional SQLite usage database at
`.dashboard-cache/usage.sqlite3` (gitignored). Existing `hours.json`,
`days.json`, and `meta.json` caches are imported once. Acquired hourly
buckets are retained indefinitely and fully covered finalised days are
rolled up transactionally.

Automatic refresh is independent of the selected display range. At or
after **00:15 UTC**, every account request inspects all 24 hours of the
previous UTC day. An hour is confirmed only when its cached aggregate
gross service cost is non-zero; missing and exactly zero-cost buckets are
unconfirmed. More than 12 unconfirmed hours triggers a fetch, so at least
12 non-zero-cost hours confirm the day. An all-zero response remains
unconfirmed.

An accepted automatic attempt records its timestamp before AWS is called.
Reloads are then throttled for 15 minutes even if the AWS request fails or
returns all zeroes. A persisted ten-minute lease coalesces concurrent
dashboard processes. Every accepted fetch makes two Cost Explorer calls
and transactionally replaces the window from the previous UTC day at
00:00 through the end of the current UTC hour: one query for EC2 usage by
instance type and one for gross cost by service. `last_fetch` is committed
only with successful data replacement.

Account plots load automatically and reload when their date or granularity
controls change. **Force fetch** is the only fetch control; it bypasses the
automatic time and content gate with an authenticated POST and has its own
persisted five-minute attempt limit. It fetches the same fixed window, not
the selected historical range. The dashboard never attempts to acquire
missing history before its retained dataset; it renders available cache
data and reports unavailable coverage. Requests may extend into the future:
data access stops at the current UTC hour while future plot buckets remain
visible as empty slots. The default view is hourly for the latest three
browser-local calendar days through the current hour, and visible absolute
timestamps use the browser's local timezone.

The local server binds only to `127.0.0.1`. It validates the exact
localhost Host and Origin, injects a per-process token into the page, and
requires that token in a custom header for every API request. Responses
disable caching and set restrictive CSP, framing, referrer, and MIME
headers. ECharts remains CDN-hosted, but its exact bytes are pinned with
Subresource Integrity and `crossorigin="anonymous"`; all dashboard
JavaScript is served locally. Missing spot-price or termination telemetry
is shown as incomplete and is never converted to a zero-cost leg.

Requirements: `gh` authenticated to the repo and the `cubie-fleet` AWS
profile; the pinned ECharts asset needs browser internet access. The AWS
CLI subprocess is forced to UTF-8 (it otherwise dies on Windows rendering
the non-breaking spaces CloudTrail events carry).
