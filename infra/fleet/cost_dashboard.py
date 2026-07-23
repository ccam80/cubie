#!/usr/bin/env python3
"""Local dashboard for RunsOn Fleet GPU CI cost and timing.

Serves an interactive page backed by a local JSON API:

  GET /api/runs                     recent ci_cuda_tests.yml runs
  GET /api/run?id=<run-id>          per-leg timing + spot cost for one run
  GET /api/account?start&end&gran   account usage/cost for a date range
  POST /api/account/refresh?...     force a Cost Explorer refresh

Per-run data is free to fetch (GitHub API + ec2:DescribeSpotPriceHistory
+ cloudtrail:LookupEvents carry no charge). Only the account panels touch
Cost Explorer, billed at $0.01 per GetCostAndUsage request. Hourly usage
is retained in a transactional SQLite store. When a requested range
extends beyond the latest non-zero hour and the last successful fetch is
at least a day old, the latest tail is fetched again, including a
12-hour settlement overlap. Force fetch bypasses that daily decision
but remains protected by a short persisted rate limit.

The loopback server validates Host and Origin, requires a per-process
token for API requests, accepts paid force refreshes only by POST, and
serves a restrictive security policy. Missing run-cost telemetry remains
unknown throughout the API and UI rather than being coerced to zero.

Run:  python infra/fleet/cost_dashboard.py   (opens http://localhost:8787)
Needs: gh authenticated to the repo, the cubie-fleet AWS profile.
"""

import argparse
import hmac
import json
import os
import re
import secrets
import sqlite3
import subprocess
import threading
import webbrowser
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

REPO = "cubiepy/cubie"
WORKFLOW = "ci_cuda_tests.yml"
PROFILE = "cubie-fleet"
REGION = "ap-southeast-2"
EC2_COMPUTE = "Amazon Elastic Compute Cloud - Compute"
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".dashboard-cache"
USAGE_DB = CACHE_DIR / "usage.sqlite3"
# Cost Explorer exposes only a recent hourly window. Once acquired, every
# hourly bucket remains in SQLite. Missing history before the first retained
# bucket remains unavailable rather than triggering a paid historical query.
CE_HOURLY_WINDOW_DAYS = 14
HOURLY_REFETCH_BACK = 12
REFRESH_THROTTLE = timedelta(days=1)
REFRESH_LEASE = timedelta(minutes=10)
FORCE_RATE_LIMIT = timedelta(minutes=5)
MAX_DAILY_RANGE_DAYS = 3660
MAX_HOURLY_RANGE_DAYS = 366
USAGE_SCHEMA_VERSION = 2
USAGE_QUERY_VERSION = "ce-usage-v1"
API_TOKEN = secrets.token_urlsafe(32)
TOKEN_HEADER = "X-Cubie-Dashboard-Token"

# The AWS CLI encodes its own stdout with the process locale; on a Windows
# cp1252 console it dies rendering the U+202F characters CloudTrail events
# carry. Force UTF-8 for every subprocess.
_ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

_RUN_CACHE = {}
_CE_LOCK = threading.Lock()


# ------------------------------------------------------------------ shells
def gh(*args):
    out = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_ENV,
    )
    if out.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed:\n{out.stderr}")
    return out.stdout


def aws(*args):
    """Return (ok, parsed-json-or-stderr). ok=False on AccessDenied."""
    out = subprocess.run(
        ["aws", *args, "--profile", PROFILE, "--output", "json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_ENV,
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
    data = json.loads(
        gh(
            "api",
            f"repos/{REPO}/actions/workflows/{WORKFLOW}/runs?per_page={limit}",
        )
    )
    runs = []
    for r in data.get("workflow_runs", []):
        # skipped runs are comment/push triggers that never launched the
        # paid GPU matrix, so they carry no legs -- drop them.
        if r["conclusion"] == "skipped":
            continue
        runs.append(
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "event": r["event"],
                "status": r["status"],
                "conclusion": r["conclusion"],
                "title": r.get("display_title", "")[:70],
            }
        )
    return runs


def fetch_legs(run_id):
    pages = json.loads(
        gh(
            "api",
            "--paginate",
            "--slurp",
            f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100",
        )
    )
    legs = []
    jobs = [job for page in pages for job in page.get("jobs", [])]
    for j in jobs:
        if not j["name"].startswith("cuda ("):
            continue
        m = re.search(r"runs-on--(i-[0-9a-f]+)--", j["runner_name"] or "")
        if not (m and j["started_at"] and j["completed_at"]):
            continue
        legs.append(
            {
                "label": j["name"][6:-1],
                "job_id": j["id"],
                "instance_id": m.group(1),
                "job_start": ts(j["started_at"]),
                "job_end": ts(j["completed_at"]),
                "steps": [
                    {
                        "name": s["name"],
                        "start": ts(s["started_at"]),
                        "end": ts(s["completed_at"]),
                    }
                    for s in j["steps"]
                    if s["started_at"] and s["completed_at"]
                ],
            }
        )
    return legs


def fetch_log(job_id):
    cache = CACHE_DIR / "logs" / f"job-{job_id}.log"
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    text = gh("api", f"repos/{REPO}/actions/jobs/{job_id}/logs").replace(
        "\r", ""
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


_BANNER = {
    "instance_type": re.compile(r"│ InstanceType\s+│ (\S+)"),
    "az": re.compile(r"│ AvailabilityZone\s+│ (\S+)"),
    "platform": re.compile(r"│ Platform\s+│ (\S+)"),
}
_PHASE = re.compile(r"│ (\d{4}-\d\d-\d\dT[\d:]+Z) │ ([a-z0-9-]+)\s+│ (\d+)ms")


def parse_log(text):
    info = {
        k: (m.group(1) if (m := rx.search(text)) else None)
        for k, rx in _BANNER.items()
    }
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
        "ec2",
        "describe-spot-price-history",
        "--region",
        REGION,
        "--instance-types",
        itype,
        "--availability-zone",
        az,
        "--product-descriptions",
        prod,
        "--start-time",
        (at - timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "--end-time",
        at.strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        "cloudtrail",
        "lookup-events",
        "--region",
        REGION,
        "--lookup-attributes",
        f"AttributeKey=ResourceName,AttributeValue={iid}",
        "--start-time",
        after.strftime("%Y-%m-%dT%H:%M:%SZ"),
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
def _billing_values(run_start, termination, price):
    """Return billed hours and cost only from complete telemetry."""
    if termination is None:
        return None, None
    billed_hours = (termination - run_start).total_seconds() / 3600
    cost = billed_hours * price if price is not None else None
    return billed_hours, cost


def _enrich_leg(leg):
    log = parse_log(fetch_log(leg["job_id"]))
    leg.update(log)
    run_start = log["running_start"] or leg["job_start"]
    term = terminate_time(leg["instance_id"], run_start)
    price = (
        spot_price(log["instance_type"], log["az"], run_start, log["platform"])
        if log["instance_type"] and log["az"]
        else None
    )
    dur_h, cost = _billing_values(run_start, term, price)
    first = leg["steps"][0]["start"] if leg["steps"] else run_start
    last = leg["steps"][-1]["end"] if leg["steps"] else leg["job_end"]
    merged = {}
    for s in leg["steps"]:
        nm = _step_name(s["name"])
        merged[nm] = (
            merged.get(nm, 0.0) + (s["end"] - s["start"]).total_seconds()
        )
    leg["_derived"] = {
        "run_start": run_start,
        "price": price,
        "cost": cost,
        "billed_hours": dur_h,
        "boot_s": max(0.0, (first - run_start).total_seconds()),
        "steps_s": max(0.0, (last - first).total_seconds()),
        "shutdown_s": (
            max(0.0, (term - last).total_seconds())
            if term is not None
            else None
        ),
        "wait_s": (log["wait_ms"] or 0) / 1000,
        "step_durs": merged,
        "termination_known": term is not None,
        "price_known": price is not None,
    }
    return leg


def run_payload(run_id):
    if run_id in _RUN_CACHE:
        return _RUN_CACHE[run_id]
    legs = fetch_legs(run_id)
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_enrich_leg, legs))
    legs.sort(key=lambda leg: leg["_derived"]["run_start"])
    t0 = min(
        (leg["job_scheduled"] or leg["_derived"]["run_start"] for leg in legs),
        default=now_utc(),
    )
    out = {"run_id": str(run_id), "t0": t0.isoformat(), "legs": []}
    for leg in legs:
        d = leg["_derived"]
        out["legs"].append(
            {
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
                "termination_known": d["termination_known"],
                "price_known": d["price_known"],
                "offset_s": (
                    (leg["job_scheduled"] or d["run_start"]) - t0
                ).total_seconds(),
                "steps": [
                    {"name": n, "dur_s": v} for n, v in d["step_durs"].items()
                ],
            }
        )
    _RUN_CACHE[run_id] = out
    return out


# ----------------------------------------------------- cost explorer cache
def _ce_query(gran, start, end, group_key, metric, service=None):
    dims = [{"Dimensions": {"Key": "RECORD_TYPE", "Values": ["Usage"]}}]
    if service:
        dims.append({"Dimensions": {"Key": "SERVICE", "Values": [service]}})
    flt = dims[0] if len(dims) == 1 else {"And": dims}
    ok, res = aws(
        "ce",
        "get-cost-and-usage",
        "--time-period",
        f"Start={start},End={end}",
        "--granularity",
        gran,
        "--metrics",
        metric,
        "--group-by",
        f"Type=DIMENSION,Key={group_key}",
        "--filter",
        json.dumps(flt),
    )
    if not ok:
        raise PermissionError(res)
    return res["ResultsByTime"]


def _validate_account_range(start, end, gran):
    """Validate a bounded account range whose two dates are inclusive."""
    gran = gran.upper()
    if gran not in {"DAILY", "HOURLY"}:
        raise ValueError("granularity must be DAILY or HOURLY")
    if not all(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) for value in (start, end)
    ):
        raise ValueError("start and end must use YYYY-MM-DD")
    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError as exc:
        raise ValueError("start and end must use YYYY-MM-DD") from exc
    if start_date > end_date:
        raise ValueError("start must be on or before end")
    days = (end_date - start_date).days + 1
    limit = MAX_HOURLY_RANGE_DAYS if gran == "HOURLY" else MAX_DAILY_RANGE_DAYS
    if days > limit:
        raise ValueError(
            f"{gran.lower()} ranges are limited to {limit} inclusive days"
        )
    return (
        start_date,
        end_date,
        end_date + timedelta(days=1),
        gran,
    )


def _expected_buckets(start, end, gran):
    """Return bucket starts for inclusive start and end dates."""
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    current = datetime.combine(
        start_date, datetime.min.time(), tzinfo=timezone.utc
    )
    stop = datetime.combine(
        end_date + timedelta(days=1),
        datetime.min.time(),
        tzinfo=timezone.utc,
    )
    step = timedelta(hours=1) if gran == "HOURLY" else timedelta(days=1)
    fmt = "%Y-%m-%dT%H:00:00Z" if gran == "HOURLY" else "%Y-%m-%d"
    out = []
    while current < stop:
        out.append((current.strftime(fmt), current))
        current += step
    return out


def _parse_ce(usage_rt, cost_rt):
    out = {}
    for rt in usage_rt:
        d = out.setdefault(
            rt["TimePeriod"]["Start"], {"usage": {}, "cost": {}}
        )
        for g in rt["Groups"]:
            t = g["Keys"][0]
            if t != "NoInstanceType":
                d["usage"][t] = float(g["Metrics"]["UsageQuantity"]["Amount"])
    for rt in cost_rt:
        d = out.setdefault(
            rt["TimePeriod"]["Start"], {"usage": {}, "cost": {}}
        )
        for g in rt["Groups"]:
            d["cost"][g["Keys"][0]] = float(
                g["Metrics"]["UnblendedCost"]["Amount"]
            )
    return out


def _total(d):
    return sum(abs(value) for value in d.get("usage", {}).values())


def _rollup(hours, day):
    u, c = {}, {}
    for h, d in hours.items():
        if h[:10] != day:
            continue
        for k, v in d.get("usage", {}).items():
            u[k] = u.get(k, 0.0) + v
        for k, v in d.get("cost", {}).items():
            c[k] = c.get(k, 0.0) + v
    return {"usage": u, "cost": c}


class UsageStore:
    """Transactional local store for retained Cost Explorer usage."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self):
        with self._connection() as connection:
            schema_version = connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0]
            if schema_version not in (0, 1, USAGE_SCHEMA_VERSION):
                raise RuntimeError(
                    "unsupported usage database schema "
                    f"{schema_version}; expected {USAGE_SCHEMA_VERSION}"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS hourly (
                    period_start TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    total REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS daily (
                    day TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    total REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            migrated = connection.execute(
                "SELECT value FROM metadata WHERE key = 'legacy_migrated'"
            ).fetchone()
            if migrated is None:
                self._migrate_legacy(connection)
                self._set_metadata(
                    connection, "legacy_migrated", now_utc().isoformat()
                )
            query_version = connection.execute(
                "SELECT value FROM metadata WHERE key = 'query_version'"
            ).fetchone()
            if query_version and query_version[0] != USAGE_QUERY_VERSION:
                raise RuntimeError(
                    "unsupported usage query version "
                    f"{query_version[0]}; expected {USAGE_QUERY_VERSION}"
                )
            self._set_metadata(
                connection, "query_version", USAGE_QUERY_VERSION
            )
            if schema_version == 1:
                for table in ("hourly", "daily"):
                    rows = connection.execute(
                        f"SELECT rowid, payload FROM {table}"
                    ).fetchall()
                    connection.executemany(
                        f"UPDATE {table} SET total = ? WHERE rowid = ?",
                        [
                            (_total(json.loads(payload)), rowid)
                            for rowid, payload in rows
                        ],
                    )
            connection.execute(f"PRAGMA user_version = {USAGE_SCHEMA_VERSION}")

    def _migrate_legacy(self, connection):
        legacy = {}
        for name in ("hours.json", "days.json", "meta.json"):
            path = self.path.parent / name
            legacy[name] = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.exists()
                else {}
            )
        for period_start, payload in legacy["hours.json"].items():
            self._upsert_bucket(
                connection, "hourly", "period_start", period_start, payload
            )
        for day, payload in legacy["days.json"].items():
            self._upsert_bucket(connection, "daily", "day", day, payload)
        for key, value in legacy["meta.json"].items():
            self._set_metadata(connection, key, str(value))

    @staticmethod
    def _set_metadata(connection, key, value):
        connection.execute(
            """
            INSERT INTO metadata(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    @staticmethod
    def _upsert_bucket(connection, table, key_column, key, payload):
        connection.execute(
            f"""
            INSERT INTO {table}({key_column}, payload, total)
            VALUES (?, ?, ?)
            ON CONFLICT({key_column}) DO UPDATE SET
                payload = excluded.payload,
                total = excluded.total
            """,
            (key, json.dumps(payload, sort_keys=True), _total(payload)),
        )

    def metadata(self, key):
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def frontier(self):
        with self._connection() as connection:
            row = connection.execute(
                "SELECT MAX(period_start) FROM hourly WHERE total > 0"
            ).fetchone()
        return row[0] if row else None

    def hourly_bounds(self):
        """Return the first and last retained hourly bucket starts."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT MIN(period_start), MAX(period_start) FROM hourly"
            ).fetchone()
        return row if row else (None, None)

    def hourly_day_counts(self, start, end):
        """Return retained hourly-bucket counts keyed by UTC day."""
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT substr(period_start, 1, 10), COUNT(*)
                FROM hourly
                WHERE period_start >= ? AND period_start < ?
                GROUP BY substr(period_start, 1, 10)
                """,
                (f"{start}T00:00:00Z", f"{end}T00:00:00Z"),
            ).fetchall()
        return dict(rows)

    def _range(self, table, key, start, end):
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT {key}, payload FROM {table}
                WHERE {key} >= ? AND {key} < ?
                ORDER BY {key}
                """,
                (start, end),
            ).fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def hourly_range(self, start, end):
        return self._range("hourly", "period_start", start, end)

    def daily_range(self, start, end):
        return self._range("daily", "day", start, end)

    def acquire_refresh_lease(self, now, force=False):
        owner = uuid4().hex
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            lease_row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'refresh_lease'"
            ).fetchone()
            if lease_row:
                lease = json.loads(lease_row[0])
                if ts(lease["until"]) > now:
                    connection.rollback()
                    return "in_progress", None
            if force:
                last_force = connection.execute(
                    """
                    SELECT value FROM metadata
                    WHERE key = 'last_force_attempt'
                    """
                ).fetchone()
                if last_force and now - ts(last_force[0]) < FORCE_RATE_LIMIT:
                    connection.rollback()
                    return "rate_limited", None
                self._set_metadata(
                    connection, "last_force_attempt", now.isoformat()
                )
            lease = {
                "owner": owner,
                "until": (now + REFRESH_LEASE).isoformat(),
            }
            self._set_metadata(connection, "refresh_lease", json.dumps(lease))
            connection.commit()
        return "acquired", owner

    def release_refresh_lease(self, owner):
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'refresh_lease'"
            ).fetchone()
            if row and json.loads(row[0]).get("owner") == owner:
                connection.execute(
                    "DELETE FROM metadata WHERE key = 'refresh_lease'"
                )

    @staticmethod
    def _delete_owned_lease(connection, owner):
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'refresh_lease'"
        ).fetchone()
        if row and json.loads(row[0]).get("owner") == owner:
            connection.execute(
                "DELETE FROM metadata WHERE key = 'refresh_lease'"
            )

    @staticmethod
    def _assert_owned_lease(connection, owner, completed_at):
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'refresh_lease'"
        ).fetchone()
        lease = json.loads(row[0]) if row else {}
        if lease.get("owner") != owner or ts(lease["until"]) <= completed_at:
            raise RuntimeError("refresh lease expired or ownership was lost")

    def commit_hourly_refresh(
        self, start, end, buckets, fetched_at, owner, force=False
    ):
        """Replace a fetched interval and rebuild its finalised days."""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_owned_lease(connection, owner, fetched_at)
            connection.execute(
                """
                DELETE FROM hourly
                WHERE period_start >= ? AND period_start < ?
                """,
                (start, end),
            )
            for period_start, payload in buckets.items():
                self._upsert_bucket(
                    connection, "hourly", "period_start", period_start, payload
                )
            frontier_row = connection.execute(
                "SELECT MAX(period_start) FROM hourly WHERE total > 0"
            ).fetchone()
            frontier = frontier_row[0] if frontier_row else None
            current_day = ts(start).date()
            last_touched_day = (ts(end) - timedelta(microseconds=1)).date()
            frontier_day = (
                date.fromisoformat(frontier[:10]) if frontier else None
            )
            while current_day <= last_touched_day:
                day = current_day.isoformat()
                next_day = (current_day + timedelta(days=1)).isoformat()
                rows = connection.execute(
                    """
                    SELECT period_start, payload FROM hourly
                    WHERE period_start >= ? AND period_start < ?
                    """,
                    (f"{day}T00:00:00Z", f"{next_day}T00:00:00Z"),
                ).fetchall()
                if (
                    frontier_day
                    and current_day < frontier_day
                    and len(rows) == 24
                ):
                    hourly = {row[0]: json.loads(row[1]) for row in rows}
                    self._upsert_bucket(
                        connection,
                        "daily",
                        "day",
                        day,
                        _rollup(hourly, day),
                    )
                else:
                    connection.execute(
                        "DELETE FROM daily WHERE day = ?", (day,)
                    )
                current_day += timedelta(days=1)
            self._set_metadata(
                connection, "last_fetch", fetched_at.isoformat()
            )
            if force:
                self._set_metadata(
                    connection, "last_force_fetch", fetched_at.isoformat()
                )
            self._delete_owned_lease(connection, owner)
            connection.commit()

    def recompute_daily(self, days):
        """Build missing finalised days when all 24 hours are retained."""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for day in days:
                day_date = date.fromisoformat(day)
                next_day = (day_date + timedelta(days=1)).isoformat()
                rows = connection.execute(
                    """
                    SELECT period_start, payload FROM hourly
                    WHERE period_start >= ? AND period_start < ?
                    """,
                    (f"{day}T00:00:00Z", f"{next_day}T00:00:00Z"),
                ).fetchall()
                if len(rows) == 24:
                    hourly = {row[0]: json.loads(row[1]) for row in rows}
                    self._upsert_bucket(
                        connection, "daily", "day", day, _rollup(hourly, day)
                    )


def _should_refresh(
    range_end,
    frontier,
    hourly_high_water,
    last_fetch,
    now,
    force=False,
):
    """Return the exact frontier/staleness refresh decision."""
    if force:
        return True
    refresh_boundary = frontier or hourly_high_water
    extends_boundary = refresh_boundary is None or range_end > ts(
        refresh_boundary
    )
    stale = last_fetch is None or now - ts(last_fetch) >= REFRESH_THROTTLE
    return extends_boundary and stale


def _complete_hourly_buckets(start, end, parsed):
    """Represent zero/missing CE hours explicitly for coverage checks."""
    current = ts(start)
    stop = ts(end)
    buckets = {}
    while current < stop:
        key = current.strftime("%Y-%m-%dT%H:00:00Z")
        buckets[key] = parsed.get(key, {"usage": {}, "cost": {}})
        current += timedelta(hours=1)
    return buckets


def _hourly_refresh_window(frontier, now):
    """Return the inclusive-start/exclusive-end CE hourly window."""
    retained_start_day = now.date() - timedelta(days=CE_HOURLY_WINDOW_DAYS - 1)
    retained_start = datetime.combine(
        retained_start_day, datetime.min.time(), tzinfo=timezone.utc
    )
    if frontier:
        start_dt = max(
            ts(frontier) - timedelta(hours=HOURLY_REFETCH_BACK),
            retained_start,
        )
    else:
        start_dt = retained_start
    end_dt = now.replace(minute=0, second=0, microsecond=0) + timedelta(
        hours=1
    )
    fmt = "%Y-%m-%dT%H:00:00Z"
    return start_dt.strftime(fmt), end_dt.strftime(fmt)


def _refresh(store, now, force=False):
    """Fetch and transactionally replace the settlement-aware hourly tail."""
    lease_status, owner = store.acquire_refresh_lease(now, force)
    if lease_status != "acquired":
        return lease_status
    frontier = store.frontier()
    start, end = _hourly_refresh_window(frontier, now)
    try:
        parsed = _parse_ce(
            _ce_query(
                "HOURLY",
                start,
                end,
                "INSTANCE_TYPE",
                "UsageQuantity",
                EC2_COMPUTE,
            ),
            _ce_query("HOURLY", start, end, "SERVICE", "UnblendedCost"),
        )
        buckets = _complete_hourly_buckets(start, end, parsed)
        store.commit_hourly_refresh(
            start, end, buckets, now_utc(), owner, force
        )
    except Exception:
        store.release_refresh_lease(owner)
        raise
    return "fetched"


def _coverage(expected, shown, gran, hourly_high_water, today=None):
    available = {key for key, _ in shown}
    missing = [key for key, _ in expected if key not in available]
    warning = None
    if missing:
        warning = (
            f"{len(missing)} of {len(expected)} requested "
            f"{gran.lower()} buckets are not available in the local "
            "usage database."
        )
    has_partial_bucket = False
    if gran == "DAILY" and hourly_high_water:
        high_water_day = hourly_high_water[:10]
        high_water_hour = ts(hourly_high_water).hour
        today = today or now_utc().date()
        has_partial_bucket = high_water_day in available and (
            high_water_hour < 23 or high_water_day == today.isoformat()
        )
        if has_partial_bucket:
            partial_message = (
                f"{high_water_day} is partial through {hourly_high_water}; "
                "Cost Explorer can revise it while billing settles."
            )
            warning = (
                f"{warning} {partial_message}" if warning else partial_message
            )
    return {
        "complete": not missing and not has_partial_bucket,
        "requested_buckets": len(expected),
        "available_buckets": len(shown),
        "missing_buckets": len(missing),
        "first_available": shown[0][0] if shown else None,
        "last_available": shown[-1][0] if shown else None,
        "warning": warning,
    }


def account_payload(start, end, gran, force=False, store=None, now=None):
    """Assemble account data for an inclusive, bounded date range.

    A normal refresh occurs exactly when the range extends beyond the
    latest non-zero hour (or the hourly high-water mark when all cached
    usage is zero) and the last successful fetch is at least one day old.
    Force bypasses that decision, while a persisted lease and five-minute
    force rate limit coalesce duplicate paid requests.
    """
    start_date, _, end_exclusive, gran = _validate_account_range(
        start, end, gran
    )
    store = store or UsageStore(USAGE_DB)
    now = now or now_utc()
    requested_start = datetime.combine(
        start_date, datetime.min.time(), tzinfo=timezone.utc
    )
    requested_end = datetime.combine(
        end_exclusive, datetime.min.time(), tzinfo=timezone.utc
    )
    current_hour_end = now.replace(
        minute=0, second=0, microsecond=0
    ) + timedelta(hours=1)
    data_end = min(requested_end, current_hour_end)
    has_data_window = requested_start < data_end
    refresh_status = "cached"
    charged = False
    with _CE_LOCK:
        frontier = store.frontier()
        _, hourly_high_water = store.hourly_bounds()
        last_fetch = store.metadata("last_fetch")
        should_refresh = force or (
            has_data_window
            and _should_refresh(
                data_end,
                frontier,
                hourly_high_water,
                last_fetch,
                now,
            )
        )
        if should_refresh:
            refresh_status = _refresh(store, now, force)
            charged = refresh_status == "fetched"

        frontier = store.frontier()
        frontier_day = frontier[:10] if frontier else None
        _, hourly_high_water = store.hourly_bounds()
        expected_days = [
            day for day, _ in _expected_buckets(start, end, "DAILY")
        ]
        cache_end_date = min(end_exclusive, now.date() + timedelta(days=1))
        stop_day = cache_end_date.isoformat()
        hourly_counts = (
            store.hourly_day_counts(start_date.isoformat(), stop_day)
            if start_date < cache_end_date
            else {}
        )
        cached_days = {}
        if gran == "DAILY" and start_date < cache_end_date:
            final_days = [
                day
                for day in expected_days
                if frontier_day
                and day < frontier_day
                and hourly_counts.get(day) == 24
            ]
            store.recompute_daily(final_days)
            cached_days = store.daily_range(start_date.isoformat(), stop_day)

        expected = _expected_buckets(start, end, gran)
        if gran == "HOURLY" and end_exclusive == now.date() + timedelta(
            days=1
        ):
            expected = [
                bucket for bucket in expected if bucket[1] < current_hour_end
            ]
        data_expected = [bucket for bucket in expected if bucket[1] < data_end]
        if gran == "HOURLY":
            hourly = (
                store.hourly_range(
                    f"{start}T00:00:00Z",
                    data_end.strftime("%Y-%m-%dT%H:00:00Z"),
                )
                if has_data_window
                else {}
            )
            shown = [
                (period, hourly[period])
                for period, _ in data_expected
                if period in hourly
            ]
        else:
            shown = []
            for day, bucket_start in data_expected:
                prefer_hourly = hourly_counts.get(day) == 24 and (
                    not frontier_day or day >= frontier_day
                )
                prefer_hourly = prefer_hourly or day == frontier_day
                prefer_hourly = prefer_hourly or (
                    hourly_high_water and day == hourly_high_water[:10]
                )
                if not prefer_hourly and day in cached_days:
                    shown.append((day, cached_days[day]))
                    continue
                bucket_end = min(bucket_start + timedelta(days=1), data_end)
                hourly = store.hourly_range(
                    f"{day}T00:00:00Z",
                    bucket_end.strftime("%Y-%m-%dT%H:00:00Z"),
                )
                if hourly:
                    shown.append((day, _rollup(hourly, day)))

    times = [key for key, _ in expected]
    shown_by_key = dict(shown)

    def series(field):
        keys = sorted(
            {name for _, payload in shown for name in payload.get(field, {})}
        )
        output = []
        for name in keys:
            row = [
                (
                    shown_by_key[period].get(field, {}).get(name, 0.0)
                    if period in shown_by_key
                    else None
                )
                for period, _ in expected
            ]
            if any(value for value in row if value is not None):
                output.append({"name": name, "data": row})
        return output

    coverage = _coverage(
        data_expected, shown, gran, hourly_high_water, now.date()
    )
    coverage["future_buckets"] = len(expected) - len(data_expected)
    return {
        "start": start,
        "end": end,
        "end_inclusive": True,
        "granularity": gran,
        "charged": charged,
        "refresh_status": refresh_status,
        "times": times,
        "frontier": frontier,
        "last_fetch": store.metadata("last_fetch"),
        "coverage": coverage,
        "usage": series("usage"),
        "cost": series("cost"),
    }


# ------------------------------------------------------------------ server
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200, ctype="application/json"):
        body = (
            obj if isinstance(obj, bytes) else json.dumps(obj).encode("utf-8")
        )
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Security-Policy",
            (
                "default-src 'none'; "
                "script-src 'self' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; connect-src 'self'; "
                "base-uri 'none'; frame-ancestors 'none'; "
                "form-action 'none'"
            ),
        )
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Permissions-Policy",
            (
                "accelerometer=(), camera=(), geolocation=(), "
                "microphone=(), payment=(), usb=()"
            ),
        )
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def _allowed_origins(self):
        port = self.server.server_address[1]
        return {
            f"http://localhost:{port}",
            f"http://127.0.0.1:{port}",
        }

    def _authorize(self, api=False, require_origin=False):
        allowed = self._allowed_origins()
        allowed_hosts = {urlparse(origin).netloc for origin in allowed}
        if self.headers.get("Host") not in allowed_hosts:
            self._send({"error": "forbidden-host"}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in allowed:
            self._send({"error": "forbidden-origin"}, 403)
            return False
        if require_origin and origin not in allowed:
            self._send({"error": "origin-required"}, 403)
            return False
        if api and not hmac.compare_digest(
            self.headers.get(TOKEN_HEADER, ""), API_TOKEN
        ):
            self._send({"error": "forbidden-token"}, 403)
            return False
        return True

    @staticmethod
    def _account_query(query):
        return (
            query["start"][0],
            query["end"][0],
            query.get("gran", ["DAILY"])[0],
        )

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if not self._authorize(api=u.path.startswith("/api/")):
                return
            if u.path in ("/", "/index.html"):
                html = (
                    (HERE / "cost_dashboard.html")
                    .read_text(encoding="utf-8")
                    .replace("__API_TOKEN__", API_TOKEN)
                    .encode("utf-8")
                )
                return self._send(html, ctype="text/html; charset=utf-8")
            if u.path == "/cost_dashboard.js":
                script = (HERE / "cost_dashboard.js").read_bytes()
                return self._send(
                    script, ctype="text/javascript; charset=utf-8"
                )
            if u.path == "/api/runs":
                return self._send(fetch_runs())
            if u.path == "/api/run":
                run_id = q["id"][0]
                if not run_id.isdecimal() or int(run_id) < 1:
                    raise ValueError("run id must be a positive integer")
                return self._send(run_payload(run_id))
            if u.path == "/api/account":
                if "force" in q:
                    return self._send(
                        {
                            "error": "method-not-allowed",
                            "detail": (
                                "force refresh requires POST "
                                "/api/account/refresh"
                            ),
                        },
                        405,
                    )
                return self._send(account_payload(*self._account_query(q)))
            return self._send({"error": "not found"}, 404)
        except (KeyError, ValueError) as exc:
            self._send({"error": "invalid-request", "detail": str(exc)}, 400)
        except PermissionError as e:
            self._send({"error": "access-denied", "detail": str(e)}, 403)
        except Exception as e:  # noqa: BLE001  surface any failure to UI
            self._send({"error": type(e).__name__, "detail": str(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if not self._authorize(api=True, require_origin=True):
                return
            if u.path != "/api/account/refresh":
                return self._send({"error": "not found"}, 404)
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length:
                self.close_connection = True
                return self._send(
                    {
                        "error": "invalid-request",
                        "detail": "refresh requests do not accept a body",
                    },
                    400,
                )
            return self._send(
                account_payload(*self._account_query(q), force=True)
            )
        except (KeyError, ValueError) as exc:
            self._send({"error": "invalid-request", "detail": str(exc)}, 400)
        except PermissionError as exc:
            self._send({"error": "access-denied", "detail": str(exc)}, 403)
        except Exception as exc:  # noqa: BLE001  surface failure to UI
            self._send({"error": type(exc).__name__, "detail": str(exc)}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    UsageStore(USAGE_DB)
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
