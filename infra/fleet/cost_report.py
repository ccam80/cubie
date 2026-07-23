#!/usr/bin/env python3
"""Cost and timeline report for a RunsOn Fleet CI run (ci_cuda_tests.yml).

Joins three data planes, keyed on the EC2 instance id that RunsOn embeds
in every GitHub runner name (runs-on--i-<id>--...):

  * GitHub Actions Jobs API  -> per-leg CI step timings + instance ids
  * the leg's "Set up job" log -> RunsOn boot timeline, instance type/AZ,
                                  launch time, running-start marker
  * AWS (profile cubie-fleet) -> achieved spot price (ec2), instance
                                 terminate time (cloudtrail), and the
                                 account 24h/30d cost/usage (cost explorer)

CloudTrail and Cost Explorer degrade gracefully: if the deployer role has
not yet been granted those reads, the affected panels render a note
instead of failing the whole report.
"""
import argparse
import base64
import io
import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend selected above)

# The AWS CLI is a Python program that encodes its own stdout with the
# process locale; on a Windows cp1252 console it dies rendering the
# U+202F etc. that CloudTrail events carry. Force UTF-8 for subprocesses.
_ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

REPO = "cubiepy/cubie"
PROFILE = "cubie-fleet"
REGION = "ap-southeast-2"
CACHE = None  # set to a Path at runtime for log caching


# ------------------------------------------------------------------ shells
def gh(*args):
    out = subprocess.run(
        ["gh", *args], capture_output=True, text=True, encoding="utf-8",
        env=_ENV,
    )
    if out.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed:\n{out.stderr}")
    return out.stdout


def aws(*args):
    """Return (ok, parsed-json-or-stderr). ok=False on AccessDenied."""
    out = subprocess.run(
        ["aws", *args, "--profile", PROFILE, "--output", "json"],
        capture_output=True, text=True, encoding="utf-8", env=_ENV,
    )
    if out.returncode != 0:
        denied = "AccessDenied" in out.stderr or "not authorized" in out.stderr
        return (False, out.stderr.strip()) if denied else (
            _raise(f"aws {' '.join(args)} failed:\n{out.stderr}")
        )
    return True, json.loads(out.stdout or "null")


def _raise(msg):
    raise RuntimeError(msg)


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# ------------------------------------------------------------------ github
def fetch_legs(run_id):
    data = json.loads(
        gh("api", f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100")
    )
    legs = []
    for j in data["jobs"]:
        if not j["name"].startswith("cuda ("):
            continue
        m = re.search(r"runs-on--(i-[0-9a-f]+)--", j["runner_name"] or "")
        legs.append({
            "label": j["name"][6:-1],           # strip "cuda (" and ")"
            "job_id": j["id"],
            "instance_id": m.group(1) if m else None,
            "job_start": ts(j["started_at"]),
            "job_end": ts(j["completed_at"]),
            "steps": [
                {"name": s["name"],
                 "start": ts(s["started_at"]),
                 "end": ts(s["completed_at"])}
                for s in j["steps"]
                if s["started_at"] and s["completed_at"]
            ],
        })
    return legs


def fetch_log(job_id):
    cache = CACHE / f"job-{job_id}.log" if CACHE else None
    if cache and cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    text = gh("api", f"repos/{REPO}/actions/jobs/{job_id}/logs").replace(
        "\r", ""
    )
    if cache:
        cache.write_text(text, encoding="utf-8")
    return text


BANNER = {
    "instance_type": re.compile(r"│ InstanceType\s+│ (\S+)"),
    "az": re.compile(r"│ AvailabilityZone\s+│ (\S+)"),
    "platform": re.compile(r"│ Platform\s+│ (\S+)"),
}
LAUNCHED = re.compile(r"RUNS_ON_INSTANCE_LAUNCHED_AT=(\S+)")
# inner columns of the RunsOn timings table:  │ <ts> │ <phase> │ <n>ms │
PHASE = re.compile(r"│ (\d{4}-\d\d-\d\dT[\d:]+Z) │ ([a-z0-9-]+)\s+│ (\d+)ms")


def parse_log(text):
    info = {}
    for key, rx in BANNER.items():
        m = rx.search(text)
        info[key] = m.group(1) if m else None
    m = LAUNCHED.search(text)
    info["launched_at"] = ts(m.group(1)) if m else None
    phases = [(ts(a), b, int(c)) for a, b, c in PHASE.findall(text)]
    info["phases"] = phases
    by = {name: (t, ms) for t, name, ms in phases}
    info["job_scheduled"] = by["job-scheduled"][0] if "job-scheduled" in by \
        else None
    # instance-pending completes when the instance is running (first
    # billed second); its DIFF is the whole capacity+scheduling wait.
    if "instance-pending" in by:
        info["running_start"], info["wait_ms"] = by["instance-pending"]
    else:
        info["running_start"], info["wait_ms"] = None, 0
    return info


# --------------------------------------------------------------------- aws
def spot_price(itype, az, at, platform):
    prod = "Windows" if (platform or "").startswith("win") else "Linux/UNIX"
    ok, res = aws(
        "ec2", "describe-spot-price-history", "--region", REGION,
        "--instance-types", itype, "--availability-zone", az,
        "--product-descriptions", prod,
        "--start-time", (at - timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "--end-time", at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    if not ok or not res or not res.get("SpotPriceHistory"):
        return None
    # the price in effect at `at` is the most recent quote at or before it
    hist = sorted(res["SpotPriceHistory"], key=lambda h: h["Timestamp"])
    price = None
    for h in hist:
        if ts(h["Timestamp"]) <= at:
            price = float(h["SpotPrice"])
    return price if price is not None else float(hist[-1]["SpotPrice"])


def terminate_time(iid, after):
    ok, res = aws(
        "cloudtrail", "lookup-events", "--region", REGION,
        "--lookup-attributes",
        f"AttributeKey=ResourceName,AttributeValue={iid}",
        "--start-time", after.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "--end-time",
        (after + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    if not ok:
        return None, res  # res carries the denial message
    for ev in res.get("Events", []):
        if ev["EventName"] in ("TerminateInstances", "BidEvictedEvent"):
            return ts(ev["EventTime"]), None
    return None, "no TerminateInstances event found in window"


def ce(granularity, start, end, group_key, metric, service=None):
    # Filter to real usage line items. The account is credit-covered, so
    # unblended cost nets to ~0 and per-service Credit line items blow out
    # the scale (negative bars); RECORD_TYPE=Usage shows the gross usage
    # the workload actually drives.
    dims = [{"Dimensions": {"Key": "RECORD_TYPE", "Values": ["Usage"]}}]
    if service:
        dims.append({"Dimensions": {"Key": "SERVICE", "Values": [service]}})
    flt = dims[0] if len(dims) == 1 else {"And": dims}
    args = [
        "ce", "get-cost-and-usage",
        "--time-period", f"Start={start},End={end}",
        "--granularity", granularity, "--metrics", metric,
        "--group-by", f"Type=DIMENSION,Key={group_key}",
        "--filter", json.dumps(flt),
    ]
    ok, res = aws(*args)
    return (res, None) if ok else (None, res)


# ----------------------------------------------------------------- derive
def enrich(legs):
    for leg in legs:
        if not leg["instance_id"]:
            continue
        log = parse_log(fetch_log(leg["job_id"]))
        leg.update(log)
        run_start = log["running_start"] or leg["job_start"]
        leg["run_start"] = run_start
        term, why = terminate_time(leg["instance_id"], run_start)
        leg["terminate"] = term
        leg["terminate_note"] = why
        leg["billed_end"] = term or leg["job_end"]
        leg["price"] = (
            spot_price(log["instance_type"], log["az"], run_start,
                       log["platform"])
            if log["instance_type"] and log["az"] else None
        )
        dur_h = (leg["billed_end"] - run_start).total_seconds() / 3600
        leg["billed_hours"] = dur_h
        leg["cost"] = dur_h * leg["price"] if leg["price"] else None
        # time bands
        first = leg["steps"][0]["start"] if leg["steps"] else run_start
        last = leg["steps"][-1]["end"] if leg["steps"] else leg["job_end"]
        leg["boot_s"] = max(0, (first - run_start).total_seconds())
        leg["steps_s"] = max(0, (last - first).total_seconds())
        leg["shutdown_s"] = max(0, (leg["billed_end"] - last).total_seconds())
        leg["wait_s"] = (log["wait_ms"] or 0) / 1000
    return [leg for leg in legs if leg["instance_id"]]


# ---------------------------------------------------------------- plotting
def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_gantt(legs, t0):
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(legs) + 1.2))
    for i, leg in enumerate(legs):
        base = (leg["job_scheduled"] or leg["run_start"] - timedelta(
            seconds=leg["wait_s"])) - t0
        b = base.total_seconds() / 60
        w, bo, st, sh = (leg["wait_s"] / 60, leg["boot_s"] / 60,
                         leg["steps_s"] / 60, leg["shutdown_s"] / 60)
        ax.barh(i, w, left=b, color="#c7ccd1", edgecolor="none")
        ax.barh(i, bo, left=b + w, color="#4c78a8")
        ax.barh(i, st, left=b + w + bo, color="#54a24b")
        ax.barh(i, sh, left=b + w + bo + st, color="#e45756")
    ax.set_yticks(range(len(legs)))
    ax.set_yticklabels([leg["label"] for leg in legs], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("minutes since first job scheduled")
    ax.set_title("Per-leg wall-clock: wait / boot / CI steps / shutdown")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               ("#c7ccd1", "#4c78a8", "#54a24b", "#e45756")]
    ax.legend(handles, ["spot wait (not billed)", "boot", "CI steps",
                        "shutdown"], ncol=4, fontsize=8,
              loc="lower center", bbox_to_anchor=(0.5, 1.06))
    return png(fig)


def fig_leg_bands(legs):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(legs))
    boot = np.array([leg["boot_s"] / 60 for leg in legs])
    step = np.array([leg["steps_s"] / 60 for leg in legs])
    shut = np.array([leg["shutdown_s"] / 60 for leg in legs])
    ax.bar(x, boot, label="boot", color="#4c78a8")
    ax.bar(x, step, bottom=boot, label="CI steps", color="#54a24b")
    ax.bar(x, shut, bottom=boot + step, label="shutdown", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels([leg["label"] for leg in legs], rotation=45, ha="right",
                       fontsize=7)
    ax.set_ylabel("minutes (billed instance time)")
    ax.set_title("Chart 1 — per-leg billed time: boot / CI steps / shutdown "
                 "(spot wait excluded)")
    ax.legend()
    return png(fig)


def _step_name(name):
    # Collapse names that differ only by matrix axis so the legend does
    # not carry a separate entry per Python version / cubie extra / action
    # SHA (e.g. "Set up Python 3.10/3.11/3.14" -> "Set up Python").
    name = re.sub(r"@[0-9a-f]{7,}", "", name)
    name = re.sub(r"\s*\(dev[\w-]*\)", "", name)
    name = re.sub(r"\s+3\.\d+\b", "", name)
    return name.strip()


def fig_leg_steps(legs):
    names, idx = [], {}
    for leg in legs:
        for s in leg["steps"]:
            nm = _step_name(s["name"])
            if nm not in idx:
                idx[nm] = len(names)
                names.append(nm)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(legs))
    bottom = np.zeros(len(legs))
    for n in names:
        vals = np.array([
            sum((s["end"] - s["start"]).total_seconds()
                for s in leg["steps"] if _step_name(s["name"]) == n) / 60
            for leg in legs
        ])
        ax.bar(x, vals, bottom=bottom, label=n[:34],
               color=cmap(idx[n] % 20))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([leg["label"] for leg in legs], rotation=45, ha="right",
                       fontsize=7)
    ax.set_ylabel("minutes")
    ax.set_title("Chart 1 (detail) — per-leg time in each CI step")
    ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1, 1))
    return png(fig)


def fig_leg_cost(legs):
    types = sorted({leg["instance_type"] for leg in legs if leg["instance_type"]})
    cmap = plt.get_cmap("tab10")
    tcol = {t: cmap(i) for i, t in enumerate(types)}
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(legs))
    costs = [leg["cost"] or 0 for leg in legs]
    ax.bar(x, costs, color=[tcol.get(leg["instance_type"], "grey")
                            for leg in legs])
    for i, leg in enumerate(legs):
        if leg["cost"] is not None and leg["price"]:
            ax.text(i, leg["cost"], f"${leg['cost']:.3f}", ha="center",
                    va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([leg["label"] for leg in legs], rotation=45, ha="right",
                       fontsize=7)
    ax.set_ylabel("USD (billed hrs x achieved spot $/hr)")
    ax.set_title("Chart 2 — per-leg AWS cost at achieved spot price")
    handles = [plt.Rectangle((0, 0), 1, 1, color=tcol[t]) for t in types]
    ax.legend(handles, types, title="instance type", fontsize=8)
    return png(fig)


def fig_by_type(legs):
    types = sorted({leg["instance_type"] for leg in legs if leg["instance_type"]})
    mins = {t: 0.0 for t in types}
    cost = {t: 0.0 for t in types}
    for leg in legs:
        t = leg["instance_type"]
        if not t:
            continue
        mins[t] += leg["billed_hours"] * 60
        cost[t] += leg["cost"] or 0
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.bar(types, [mins[t] for t in types], color="#4c78a8")
    a1.set_ylabel("billed minutes")
    a1.set_title("Chart 3a — minutes per instance type")
    a2.bar(types, [cost[t] for t in types], color="#54a24b")
    a2.set_ylabel("USD")
    a2.set_title("Chart 3b — cost per instance type")
    for a in (a1, a2):
        a.tick_params(axis="x", rotation=20)
    return png(fig)


def fig_run_aggregate(legs):
    boot = sum(leg["boot_s"] for leg in legs) / 60
    step = sum(leg["steps_s"] for leg in legs) / 60
    shut = sum(leg["shutdown_s"] for leg in legs) / 60
    wait = sum(leg["wait_s"] for leg in legs) / 60
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4),
                                 gridspec_kw={"width_ratios": [1, 1]})
    a1.bar(["run"], [boot], label="boot", color="#4c78a8")
    a1.bar(["run"], [step], bottom=[boot], label="CI steps", color="#54a24b")
    a1.bar(["run"], [shut], bottom=[boot + step], label="shutdown",
           color="#e45756")
    a1.set_ylabel("summed billed minutes over all legs")
    a1.set_title("Chart 4 — run aggregate: boot / CI steps / shutdown")
    a1.legend()
    a2.bar(["spot wait\n(not billed)", "billed\nruntime"],
           [wait, boot + step + shut], color=["#c7ccd1", "#4c78a8"])
    a2.set_ylabel("summed minutes over all legs")
    a2.set_title("Wait vs billed runtime (separated, per your ask)")
    return png(fig)


def fig_wait(legs):
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(legs))
    ax.bar(x, [leg["wait_s"] / 60 for leg in legs], color="#c7ccd1",
           edgecolor="#8a9197")
    ax.set_xticks(x)
    ax.set_xticklabels([leg["label"] for leg in legs], rotation=45, ha="right",
                       fontsize=7)
    ax.set_ylabel("minutes")
    ax.set_title("Chart 5 (separate) — spot-capacity wait per leg "
                 "(job scheduled -> instance running; NOT billed)")
    return png(fig)


def fig_ce_stack(res, title, unit):
    if res is None:
        return None
    buckets = res["ResultsByTime"]
    # NoInstanceType is EC2-Other (EBS volume-hours, data transfer) with no
    # instance type; drop it so a "per instance type" chart is not swamped
    # by non-compute usage.
    keys = sorted({g["Keys"][0] for b in buckets for g in b["Groups"]}
                  - {"NoInstanceType"})
    times = [b["TimePeriod"]["Start"] for b in buckets]
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(12, 4.2))
    bottom = np.zeros(len(buckets))
    drawn = 0
    for i, k in enumerate(keys):
        vals = np.array([
            next((float(g["Metrics"][list(g["Metrics"])[0]]["Amount"])
                  for g in b["Groups"] if g["Keys"][0] == k), 0.0)
            for b in buckets
        ])
        if vals.sum() == 0:
            continue
        ax.bar(range(len(buckets)), vals, bottom=bottom, label=k[:26],
               color=cmap(i % 20))
        bottom += vals
        drawn += 1
    if not drawn:
        # Empty window: let the caller render an explanatory note instead
        # of a blank chart (Cost Explorer finalises recent data with lag).
        plt.close(fig)
        return None
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels([t[5:] for t in times], rotation=90, fontsize=6)
    ax.set_ylabel(unit)
    ax.set_title(title)
    ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1, 1))
    return png(fig)


# ------------------------------------------------------------------- html
def img(b64, note=None):
    if b64 is None:
        return f'<div class="missing">{note}</div>'
    return f'<img src="data:image/png;base64,{b64}">'


def money(v):
    return "" if v is None else f"${v:.4f}"


def build_html(run_id, legs, panels, meta):
    total_cost = sum(leg["cost"] or 0 for leg in legs)
    priced = sum(1 for leg in legs if leg["cost"] is not None)
    rows = "".join(
        f"<tr><td>{leg['label']}</td><td>{leg['instance_id']}</td>"
        f"<td>{leg['instance_type']}</td><td>{leg['az']}</td>"
        f"<td>{leg['wait_s']/60:.1f}</td><td>{leg['boot_s']/60:.1f}</td>"
        f"<td>{leg['steps_s']/60:.1f}</td><td>{leg['shutdown_s']/60:.1f}</td>"
        f"<td>{money(leg['price'])}</td>"
        f"<td>{money(leg['cost'])}</td></tr>"
        for leg in legs
    )
    sections = "".join(
        f'<h2>{t}</h2>{img(b, n)}' for t, b, n in panels
    )
    return f"""<!doctype html><meta charset=utf-8>
<title>Fleet CI cost report — run {run_id}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;max-width:1200px;margin:2rem auto;
      padding:0 1rem;color:#1a1a1a}}
 h1{{margin-bottom:.2rem}} h2{{margin-top:2rem;font-size:1.05rem}}
 img{{max-width:100%;border:1px solid #e5e5e5;border-radius:6px}}
 table{{border-collapse:collapse;font-size:12px;margin:.5rem 0}}
 td,th{{border:1px solid #ddd;padding:3px 7px;text-align:right}}
 td:first-child,th:first-child{{text-align:left}}
 .missing{{background:#fff6e5;border:1px solid #f0c36d;border-radius:6px;
          padding:1rem;color:#7a5b12}}
 .meta{{color:#666;font-size:12px}}
 code{{background:#f2f2f2;padding:1px 4px;border-radius:3px}}
</style>
<h1>RunsOn Fleet CI — cost &amp; timeline</h1>
<p class=meta>run <code>{run_id}</code> &middot; {len(legs)} GPU legs &middot;
 compute cost (achieved spot) <b>${total_cost:.3f}</b>
 across {priced}/{len(legs)} priced legs &middot; region {REGION}<br>{meta}</p>
<table><tr><th>leg</th><th>instance</th><th>type</th><th>AZ</th>
<th>wait m</th><th>boot m</th><th>steps m</th><th>shut m</th>
<th>spot $/h</th><th>cost $</th></tr>{rows}</table>
{sections}
"""


# ------------------------------------------------------------------- main
def main():
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--out", default="fleet_cost_report.html")
    ap.add_argument("--cache-dir", default=".")
    args = ap.parse_args()
    from pathlib import Path
    CACHE = Path(args.cache_dir)

    print(f"fetching legs for run {args.run_id} ...")
    legs = enrich(fetch_legs(args.run_id))
    legs.sort(key=lambda leg: leg["run_start"])
    print(f"  {len(legs)} GPU legs")
    t0 = min(leg["job_scheduled"] or leg["run_start"] for leg in legs)

    # AWS-side aggregate panels (24h hourly, 30d daily)
    now = datetime.now(timezone.utc)
    d1 = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    dnow = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    day30 = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    today = now.strftime("%Y-%m-%d")
    EC2 = "Amazon Elastic Compute Cloud - Compute"
    h_use, e1 = ce("HOURLY", d1, dnow, "INSTANCE_TYPE", "UsageQuantity", EC2)
    h_svc, e2 = ce("HOURLY", d1, dnow, "SERVICE", "UnblendedCost")
    d_use, e3 = ce("DAILY", day30, today, "INSTANCE_TYPE", "UsageQuantity",
                   EC2)
    d_svc, e4 = ce("DAILY", day30, today, "SERVICE", "UnblendedCost")

    lag_note = ("No finalized Cost Explorer data in this window. CE "
                "finalises recent usage with a ~24-48h lag, so a "
                "just-completed run may not appear yet -- charts 1-5 carry "
                "the run's own EC2 cost from spot price x duration. (If the "
                "deployer role lacks ce:GetCostAndUsage, the panel errors "
                "instead.)")

    def ce_panel(res, err, title, unit):
        b = fig_ce_stack(res, title, unit)
        return (b, None) if b is not None else (None, err or lag_note)

    ct_denied = next((leg["terminate_note"] for leg in legs
                      if leg["terminate_note"] and "authorized" in
                      (leg["terminate_note"] or "")), None)
    meta = []
    if ct_denied:
        meta.append("shutdown band + billed end use job-completion time "
                    "(CloudTrail LookupEvents not yet granted, so exact "
                    "terminate time is unavailable).")
    else:
        meta.append("shutdown band + billed end use the CloudTrail "
                    "TerminateInstances event.")
    meta_html = " ".join(meta)

    panels = [
        ("Overview timeline (wait / boot / CI steps / shutdown)",
         fig_gantt(legs, t0), None),
        ("Chart 1 — per-leg billed bands", fig_leg_bands(legs), None),
        ("Chart 1 detail — per-leg time in each CI step",
         fig_leg_steps(legs), None),
        ("Chart 2 — per-leg cost at achieved spot price",
         fig_leg_cost(legs), None),
        ("Chart 3 — minutes and cost per instance type",
         fig_by_type(legs), None),
        ("Chart 4 — run aggregate bands", fig_run_aggregate(legs), None),
        ("Chart 5 — spot-capacity wait per leg (separate; not billed)",
         fig_wait(legs), None),
        ("Chart 6 — last 24h, EC2 usage hrs per instance type (hourly)",
         *ce_panel(h_use, e1,
                   "EC2 usage hours per instance type — last 24h", "hours")),
        ("Chart 6 — last 24h, gross usage $ by service (hourly)",
         *ce_panel(h_svc, e2, "Gross usage $ by service — last 24h", "USD")),
        ("Chart 7 — last 30d, EC2 usage hrs per instance type (daily)",
         *ce_panel(d_use, e3,
                   "EC2 usage hours per instance type — last 30d", "hours")),
        ("Chart 7 — last 30d, gross usage $ by service (daily)",
         *ce_panel(d_svc, e4, "Gross usage $ by service — last 30d", "USD")),
    ]
    Path(args.out).write_text(build_html(args.run_id, legs, panels,
                                         meta_html), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
