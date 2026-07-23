#!/usr/bin/env python3
"""Local dashboard for RunsOn Fleet GPU CI cost and timing.

Serves an interactive page (cost_dashboard.html) backed by a small JSON
API that fetches on demand:

  GET /api/runs                     recent ci_cuda_tests.yml runs
  GET /api/run?id=<run-id>          per-leg timing + spot cost for one run
  GET /api/account?start&end&gran   account usage/cost for a date range

Per-run data is free to fetch (GitHub API + ec2:DescribeSpotPriceHistory
+ cloudtrail:LookupEvents carry no charge). Only the account panels touch
Cost Explorer, billed at $0.01 per GetCostAndUsage request; results are
cached at bucket granularity on disk and a bucket is re-fetched only when
it is missing, or still within Cost Explorer's ~2-day finalisation window
and more than a day stale. Finalised buckets are never re-fetched.

Run:  python infra/fleet/cost_dashboard.py   (opens http://localhost:8787)
Needs: gh authenticated to the repo, the cubie-fleet AWS profile.
"""
import argparse
import json
import os
import re
import subprocess
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = "cubiepy/cubie"
WORKFLOW = "ci_cuda_tests.yml"
PROFILE = "cubie-fleet"
REGION = "ap-southeast-2"
EC2_COMPUTE = "Amazon Elastic Compute Cloud - Compute"
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".dashboard-cache"

# The AWS CLI encodes its own stdout with the process locale; on a Windows
# cp1252 console it dies rendering the U+202F characters CloudTrail events
# carry. Force UTF-8 for every subprocess.
_ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

_RUN_CACHE = {}
_CE_LOCK = threading.Lock()


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
        msg = out.stderr
        if "AccessDenied" in msg or "not authorized" in msg:
            return False, msg.strip()
        raise RuntimeError(f"aws {' '.join(args)} failed:\n{msg}")
    return True, json.loads(out.stdout or "null")


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def now_utc():
    return datetime.now(timezone.utc)


def _step_name(name):
    # Collapse names differing only by matrix axis (Python version, cubie
    # extra, action SHA) so a step is one series, not one per leg flavour.
    name = re.sub(r"@[0-9a-f]{7,}", "", name)
    name = re.sub(r"\s*\(dev[\w-]*\)", "", name)
    name = re.sub(r"\s+3\.\d+\b", "", name)
    return name.strip()


# ------------------------------------------------------------------ github
def fetch_runs(limit=40):
    data = json.loads(gh(
        "api",
        f"repos/{REPO}/actions/workflows/{WORKFLOW}/runs?per_page={limit}",
    ))
    runs = []
    for r in data.get("workflow_runs", []):
        # skipped runs are comment/push triggers that never launched the
        # paid GPU matrix, so they carry no legs -- drop them.
        if r["conclusion"] == "skipped":
            continue
        runs.append({
            "id": r["id"],
            "created_at": r["created_at"],
            "event": r["event"],
            "status": r["status"],
            "conclusion": r["conclusion"],
            "title": r.get("display_title", "")[:70],
        })
    return runs


def fetch_legs(run_id):
    data = json.loads(gh(
        "api", f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100"))
    legs = []
    for j in data["jobs"]:
        if not j["name"].startswith("cuda ("):
            continue
        m = re.search(r"runs-on--(i-[0-9a-f]+)--", j["runner_name"] or "")
        if not (m and j["started_at"] and j["completed_at"]):
            continue
        legs.append({
            "label": j["name"][6:-1],
            "job_id": j["id"],
            "instance_id": m.group(1),
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
    cache = CACHE_DIR / "logs" / f"job-{job_id}.log"
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    text = gh("api", f"repos/{REPO}/actions/jobs/{job_id}/logs").replace(
        "\r", "")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


_BANNER = {
    "instance_type": re.compile(r"│ InstanceType\s+│ (\S+)"),
    "az": re.compile(r"│ AvailabilityZone\s+│ (\S+)"),
    "platform": re.compile(r"│ Platform\s+│ (\S+)"),
}
_PHASE = re.compile(
    r"│ (\d{4}-\d\d-\d\dT[\d:]+Z) │ ([a-z0-9-]+)\s+│ (\d+)ms")


def parse_log(text):
    info = {k: (m.group(1) if (m := rx.search(text)) else None)
            for k, rx in _BANNER.items()}
    phases = [(ts(a), b, int(c)) for a, b, c in _PHASE.findall(text)]
    by = {name: (t, ms) for t, name, ms in phases}
    info["job_scheduled"] = by.get("job-scheduled", (None,))[0]
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
        "--start-time",
        (at - timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "--end-time", at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    if not ok or not res or not res.get("SpotPriceHistory"):
        return None
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
        return None
    for ev in res.get("Events", []):
        if ev["EventName"] in ("TerminateInstances", "BidEvictedEvent"):
            return ts(ev["EventTime"])
    return None


# ----------------------------------------------------------------- derive
def _enrich_leg(leg):
    log = parse_log(fetch_log(leg["job_id"]))
    leg.update(log)
    run_start = log["running_start"] or leg["job_start"]
    term = terminate_time(leg["instance_id"], run_start)
    billed_end = term or leg["job_end"]
    price = (spot_price(log["instance_type"], log["az"], run_start,
                        log["platform"])
             if log["instance_type"] and log["az"] else None)
    dur_h = (billed_end - run_start).total_seconds() / 3600
    first = leg["steps"][0]["start"] if leg["steps"] else run_start
    last = leg["steps"][-1]["end"] if leg["steps"] else leg["job_end"]
    merged = {}
    for s in leg["steps"]:
        nm = _step_name(s["name"])
        merged[nm] = merged.get(nm, 0.0) + (
            s["end"] - s["start"]).total_seconds()
    leg["_derived"] = {
        "run_start": run_start,
        "price": price,
        "cost": dur_h * price if price else None,
        "billed_hours": dur_h,
        "boot_s": max(0.0, (first - run_start).total_seconds()),
        "steps_s": max(0.0, (last - first).total_seconds()),
        "shutdown_s": max(0.0, (billed_end - last).total_seconds()),
        "wait_s": (log["wait_ms"] or 0) / 1000,
        "step_durs": merged,
    }
    return leg


def run_payload(run_id):
    if run_id in _RUN_CACHE:
        return _RUN_CACHE[run_id]
    legs = fetch_legs(run_id)
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_enrich_leg, legs))
    legs.sort(key=lambda leg: leg["_derived"]["run_start"])
    t0 = min((leg["job_scheduled"] or leg["_derived"]["run_start"]
              for leg in legs), default=now_utc())
    out = {"run_id": str(run_id), "t0": t0.isoformat(), "legs": []}
    for leg in legs:
        d = leg["_derived"]
        out["legs"].append({
            "label": leg["label"],
            "instance_id": leg["instance_id"],
            "type": leg["instance_type"],
            "az": leg["az"],
            "price": d["price"],
            "cost": d["cost"],
            "billed_hours": d["billed_hours"],
            "wait_s": d["wait_s"],
            "boot_s": d["boot_s"],
            "steps_s": d["steps_s"],
            "shutdown_s": d["shutdown_s"],
            "offset_s": ((leg["job_scheduled"] or d["run_start"])
                         - t0).total_seconds(),
            "steps": [{"name": n, "dur_s": v}
                      for n, v in d["step_durs"].items()],
        })
    _RUN_CACHE[run_id] = out
    return out


# ----------------------------------------------------- cost explorer cache
def _ce_query(gran, start, end, group_key, metric, service=None):
    dims = [{"Dimensions": {"Key": "RECORD_TYPE", "Values": ["Usage"]}}]
    if service:
        dims.append({"Dimensions": {"Key": "SERVICE", "Values": [service]}})
    flt = dims[0] if len(dims) == 1 else {"And": dims}
    ok, res = aws(
        "ce", "get-cost-and-usage",
        "--time-period", f"Start={start},End={end}",
        "--granularity", gran, "--metrics", metric,
        "--group-by", f"Type=DIMENSION,Key={group_key}",
        "--filter", json.dumps(flt),
    )
    if not ok:
        raise PermissionError(res)
    return res["ResultsByTime"]


def _expected_buckets(start, end, gran):
    s = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    step = timedelta(hours=1) if gran == "HOURLY" else timedelta(days=1)
    fmt = "%Y-%m-%dT%H:00:00Z" if gran == "HOURLY" else "%Y-%m-%d"
    out, t = [], s
    while t < e:
        out.append((t.strftime(fmt), t))
        t += step
    return out


def _cache_path(gran):
    return CACHE_DIR / f"ce_{gran.lower()}.json"


def _load_ce_cache(gran):
    p = _cache_path(gran)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _save_ce_cache(gran, cache):
    p = _cache_path(gran)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache), encoding="utf-8")


def account_payload(start, end, gran):
    """Assemble usage-by-type and gross-cost-by-service for a date range,
    fetching from Cost Explorer only the buckets that are missing or still
    settling and stale."""
    gran = gran.upper()
    with _CE_LOCK:
        cache = _load_ce_cache(gran)
        expected = _expected_buckets(start, end, gran)
        now = now_utc()
        stale = []
        for key, when in expected:
            hit = cache.get(key)
            finalized = when < now - timedelta(days=2)
            if hit is None:
                stale.append(key)
            elif not finalized:
                age = now - ts(hit["fetched"])
                if age > timedelta(days=1):
                    stale.append(key)
        charged = False
        if stale:
            usage = _ce_query(gran, start, end, "INSTANCE_TYPE",
                              "UsageQuantity", EC2_COMPUTE)
            cost = _ce_query(gran, start, end, "SERVICE", "UnblendedCost")
            fetched = now.isoformat()
            merged = {}
            for rt in usage:
                key = rt["TimePeriod"]["Start"]
                d = merged.setdefault(key, {"usage": {}, "cost": {}})
                for g in rt["Groups"]:
                    t = g["Keys"][0]
                    if t == "NoInstanceType":
                        continue
                    d["usage"][t] = float(
                        g["Metrics"]["UsageQuantity"]["Amount"])
            for rt in cost:
                key = rt["TimePeriod"]["Start"]
                d = merged.setdefault(key, {"usage": {}, "cost": {}})
                for g in rt["Groups"]:
                    d["cost"][g["Keys"][0]] = float(
                        g["Metrics"]["UnblendedCost"]["Amount"])
            for key, d in merged.items():
                cache[key] = {"fetched": fetched, **d}
            _save_ce_cache(gran, cache)
            charged = True

    times, usage_keys, cost_keys = [], set(), set()
    for key, _ in expected:
        hit = cache.get(key, {"usage": {}, "cost": {}})
        times.append(key)
        usage_keys |= set(hit.get("usage", {}))
        cost_keys |= set(hit.get("cost", {}))

    def series(keys, field):
        out = []
        for k in sorted(keys):
            row = [cache.get(key, {}).get(field, {}).get(k, 0.0)
                   for key, _ in expected]
            if any(row):
                out.append({"name": k, "data": row})
        return out

    return {
        "start": start, "end": end, "granularity": gran,
        "charged": charged, "times": times,
        "usage": series(usage_keys, "usage"),
        "cost": series(cost_keys, "cost"),
    }


# ------------------------------------------------------------------ server
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200, ctype="application/json"):
        body = (obj if isinstance(obj, bytes)
                else json.dumps(obj).encode("utf-8"))
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                html = (HERE / "cost_dashboard.html").read_bytes()
                return self._send(html, ctype="text/html; charset=utf-8")
            if u.path == "/api/runs":
                return self._send(fetch_runs())
            if u.path == "/api/run":
                return self._send(run_payload(q["id"][0]))
            if u.path == "/api/account":
                return self._send(account_payload(
                    q["start"][0], q["end"][0],
                    q.get("gran", ["DAILY"])[0]))
            self._send({"error": "not found"}, 404)
        except PermissionError as e:
            self._send({"error": "access-denied", "detail": str(e)}, 403)
        except Exception as e:  # noqa: BLE001  surface any failure to UI
            self._send({"error": type(e).__name__, "detail": str(e)}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url = f"http://localhost:{args.port}"
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"fleet cost dashboard on {url}  (Ctrl-C to stop)")
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
