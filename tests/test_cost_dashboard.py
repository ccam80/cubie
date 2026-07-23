import http.client
import json
import threading
from datetime import date, datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from infra.fleet import cost_dashboard


def _payload(usage=0.0, cost=0.0):
    return {
        "usage": {"g5.xlarge": usage} if usage else {},
        "cost": {"Amazon EC2": cost} if cost else {},
    }


def _hourly(start, end, usage=0.0, cost=0.0):
    current = cost_dashboard.ts(start)
    stop = cost_dashboard.ts(end)
    result = {}
    while current < stop:
        key = current.strftime("%Y-%m-%dT%H:00:00Z")
        result[key] = _payload(usage, cost)
        current += timedelta(hours=1)
    return result


def test_account_range_is_inclusive_and_bounded():
    validated = cost_dashboard._validate_account_range(
        "2026-07-01",
        "2026-07-02",
        "daily",
        today=date(2026, 7, 2),
    )

    assert validated[:3] == (
        date(2026, 7, 1),
        date(2026, 7, 2),
        date(2026, 7, 3),
    )
    assert (
        len(
            cost_dashboard._expected_buckets(
                "2026-07-01", "2026-07-02", "DAILY"
            )
        )
        == 2
    )
    assert (
        len(
            cost_dashboard._expected_buckets(
                "2026-07-01", "2026-07-02", "HOURLY"
            )
        )
        == 48
    )

    with pytest.raises(ValueError, match="DAILY or HOURLY"):
        cost_dashboard._validate_account_range(
            "2026-07-01",
            "2026-07-02",
            "monthly",
            today=date(2026, 7, 2),
        )
    with pytest.raises(ValueError, match="on or before"):
        cost_dashboard._validate_account_range(
            "2026-07-02",
            "2026-07-01",
            "DAILY",
            today=date(2026, 7, 2),
        )
    with pytest.raises(ValueError, match="future"):
        cost_dashboard._validate_account_range(
            "2026-07-01",
            "2026-07-03",
            "DAILY",
            today=date(2026, 7, 2),
        )
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        cost_dashboard._validate_account_range(
            "20260701",
            "2026-07-02",
            "DAILY",
            today=date(2026, 7, 2),
        )
    with pytest.raises(ValueError, match="366 inclusive days"):
        cost_dashboard._validate_account_range(
            "2025-07-01",
            "2026-07-02",
            "HOURLY",
            today=date(2026, 7, 2),
        )


def test_refresh_decision_requires_frontier_extension_and_one_day_age():
    now = datetime(2026, 7, 23, 12, tzinfo=timezone.utc)
    frontier = "2026-07-22T10:00:00Z"
    beyond = datetime(2026, 7, 23, tzinfo=timezone.utc)
    before = datetime(2026, 7, 22, 9, tzinfo=timezone.utc)

    assert not cost_dashboard._should_refresh(
        before,
        frontier,
        (now - timedelta(days=2)).isoformat(),
        now,
    )
    assert not cost_dashboard._should_refresh(
        beyond,
        frontier,
        (now - timedelta(hours=23, minutes=59)).isoformat(),
        now,
    )
    assert cost_dashboard._should_refresh(
        beyond,
        frontier,
        (now - timedelta(days=1)).isoformat(),
        now,
    )
    assert not cost_dashboard._should_refresh(
        beyond,
        None,
        (now - timedelta(hours=1)).isoformat(),
        now,
    )
    assert cost_dashboard._should_refresh(
        before, frontier, now.isoformat(), now, force=True
    )


def test_refresh_window_starts_twelve_hours_behind_frontier():
    now = datetime(2026, 7, 23, 12, 34, 56, tzinfo=timezone.utc)

    assert cost_dashboard._hourly_refresh_window(
        "2026-07-22T10:00:00Z", now
    ) == (
        "2026-07-21T22:00:00Z",
        "2026-07-23T13:00:00Z",
    )


def test_refresh_window_clamps_old_frontier_to_hourly_retention():
    now = datetime(2026, 7, 23, 12, 34, 56, tzinfo=timezone.utc)

    assert cost_dashboard._hourly_refresh_window(
        "2026-06-01T10:00:00Z", now
    ) == (
        "2026-07-10T00:00:00Z",
        "2026-07-23T13:00:00Z",
    )


def test_billing_values_preserve_unknown_termination_and_price():
    start = datetime(2026, 7, 23, 10, tzinfo=timezone.utc)
    end = start + timedelta(hours=2)

    assert cost_dashboard._billing_values(start, None, 0.5) == (None, None)
    assert cost_dashboard._billing_values(start, end, None) == (2.0, None)
    assert cost_dashboard._billing_values(start, end, 0.0) == (2.0, 0.0)
    assert cost_dashboard._billing_values(start, end, 0.5) == (2.0, 1.0)


def test_dashboard_javascript_does_not_coerce_unknown_cost_to_zero():
    javascript = (
        Path(cost_dashboard.__file__)
        .with_name("cost_dashboard.js")
        .read_text(encoding="utf-8")
    )

    assert "value: leg.cost" in javascript
    assert "leg.cost ||" not in javascript


def test_usage_store_migrates_legacy_json_once(tmp_path):
    hour = "2026-07-20T12:00:00Z"
    day = "2026-07-19"
    last_fetch = "2026-07-21T00:00:00+00:00"
    (tmp_path / "hours.json").write_text(
        json.dumps({hour: _payload(1.5, 2.5)}),
        encoding="utf-8",
    )
    (tmp_path / "days.json").write_text(
        json.dumps({day: _payload(24.0, 12.0)}),
        encoding="utf-8",
    )
    (tmp_path / "meta.json").write_text(
        json.dumps({"last_fetch": last_fetch}),
        encoding="utf-8",
    )

    store = cost_dashboard.UsageStore(tmp_path / "usage.sqlite3")

    assert store.frontier() == hour
    assert store.metadata("last_fetch") == last_fetch
    assert store.hourly_range(
        "2026-07-20T00:00:00Z", "2026-07-21T00:00:00Z"
    ) == {hour: _payload(1.5, 2.5)}
    assert store.daily_range("2026-07-19", "2026-07-20") == {
        day: _payload(24.0, 12.0)
    }

    (tmp_path / "hours.json").write_text("{}", encoding="utf-8")
    reopened = cost_dashboard.UsageStore(tmp_path / "usage.sqlite3")
    assert reopened.frontier() == hour
    assert (
        reopened.metadata("query_version")
        == cost_dashboard.USAGE_QUERY_VERSION
    )
    with reopened._connection() as connection:
        assert (
            connection.execute("PRAGMA user_version").fetchone()[0]
            == cost_dashboard.USAGE_SCHEMA_VERSION
        )


def test_settlement_overlap_replaces_hours_and_existing_daily_rollup(
    tmp_path,
):
    store = cost_dashboard.UsageStore(tmp_path / "usage.sqlite3")
    first_at = datetime(2026, 7, 3, 2, tzinfo=timezone.utc)
    status, owner = store.acquire_refresh_lease(first_at)
    assert status == "acquired"
    first = _hourly(
        "2026-07-01T00:00:00Z",
        "2026-07-02T01:00:00Z",
        usage=1.0,
        cost=2.0,
    )
    store.commit_hourly_refresh(
        "2026-07-01T00:00:00Z",
        "2026-07-02T01:00:00Z",
        first,
        first_at,
        owner,
    )
    assert store.daily_range("2026-07-01", "2026-07-02")[
        "2026-07-01"
    ] == _payload(24.0, 48.0)

    second_at = first_at + timedelta(days=1)
    status, owner = store.acquire_refresh_lease(second_at)
    assert status == "acquired"
    corrected = _hourly(
        "2026-07-01T12:00:00Z",
        "2026-07-02T02:00:00Z",
        usage=2.0,
        cost=4.0,
    )
    corrected["2026-07-01T13:00:00Z"] = _payload()
    store.commit_hourly_refresh(
        "2026-07-01T12:00:00Z",
        "2026-07-02T02:00:00Z",
        corrected,
        second_at,
        owner,
    )

    daily = store.daily_range("2026-07-01", "2026-07-02")["2026-07-01"]
    assert daily == _payload(34.0, 68.0)
    assert store.metadata("last_fetch") == second_at.isoformat()
    retained = store.hourly_range(
        "2026-07-01T00:00:00Z",
        "2026-07-02T02:00:00Z",
    )
    assert len(retained) == 26
    assert retained["2026-07-01T00:00:00Z"] == _payload(1.0, 2.0)
    assert retained["2026-07-01T13:00:00Z"] == _payload()


def test_settlement_can_move_frontier_backward_without_stale_daily(tmp_path):
    store = cost_dashboard.UsageStore(tmp_path / "usage.sqlite3")
    first_at = datetime(2026, 7, 3, 2, tzinfo=timezone.utc)
    status, owner = store.acquire_refresh_lease(first_at)
    assert status == "acquired"
    store.commit_hourly_refresh(
        "2026-07-01T00:00:00Z",
        "2026-07-03T01:00:00Z",
        _hourly(
            "2026-07-01T00:00:00Z",
            "2026-07-03T01:00:00Z",
            usage=1.0,
        ),
        first_at,
        owner,
    )
    assert store.daily_range("2026-07-02", "2026-07-03")[
        "2026-07-02"
    ] == _payload(24.0)

    second_at = first_at + timedelta(hours=1)
    status, owner = store.acquire_refresh_lease(second_at)
    assert status == "acquired"
    store.commit_hourly_refresh(
        "2026-07-02T12:00:00Z",
        "2026-07-03T01:00:00Z",
        _hourly("2026-07-02T12:00:00Z", "2026-07-03T01:00:00Z"),
        second_at,
        owner,
    )

    assert store.frontier() == "2026-07-02T11:00:00Z"
    assert store.daily_range("2026-07-02", "2026-07-03") == {}
    payload = cost_dashboard.account_payload(
        "2026-07-02",
        "2026-07-02",
        "DAILY",
        store=store,
        now=second_at,
    )
    assert payload["times"] == ["2026-07-02"]
    assert payload["usage"] == [{"name": "g5.xlarge", "data": [12.0]}]


def test_zero_frontier_falls_back_only_before_recent_hourly_tail():
    expected = [
        day
        for day, _ in cost_dashboard._expected_buckets(
            "2026-07-01", "2026-07-16", "DAILY"
        )
    ]
    hourly_counts = {f"2026-07-{day:02d}": 24 for day in range(15, 17)}

    assert cost_dashboard._daily_fallback_days(
        expected,
        hourly_counts,
        frontier=None,
        tail_start="2026-07-15T00:00:00Z",
        today=date(2026, 7, 16),
    ) == [f"2026-07-{day:02d}" for day in range(1, 15)]


def test_daily_fallback_covers_gap_before_clamped_recent_tail():
    expected = [
        day
        for day, _ in cost_dashboard._expected_buckets(
            "2026-06-01", "2026-07-11", "DAILY"
        )
    ]
    hourly_counts = {
        "2026-06-01": 24,
        "2026-07-10": 24,
        "2026-07-11": 24,
    }

    fallback = cost_dashboard._daily_fallback_days(
        expected,
        hourly_counts,
        frontier="2026-06-01T12:00:00Z",
        tail_start="2026-07-10T00:00:00Z",
        today=date(2026, 7, 11),
    )

    assert fallback[0] == "2026-06-02"
    assert fallback[-1] == "2026-07-09"
    assert len(fallback) == 38


def test_usage_store_releases_sqlite_file_handles(tmp_path):
    database = tmp_path / "usage.sqlite3"
    store = cost_dashboard.UsageStore(database)

    assert store.metadata("last_fetch") is None
    assert store.hourly_bounds() == (None, None)
    database.unlink()

    assert not database.exists()


def test_frontier_uses_latest_nonzero_usage_without_cancellation(tmp_path):
    database = tmp_path / "usage.sqlite3"
    store = cost_dashboard.UsageStore(database)
    now = datetime(2026, 7, 23, 12, tzinfo=timezone.utc)
    status, owner = store.acquire_refresh_lease(now)
    assert status == "acquired"
    store.commit_hourly_refresh(
        "2026-07-23T10:00:00Z",
        "2026-07-23T12:00:00Z",
        {
            "2026-07-23T10:00:00Z": {
                "usage": {"positive": 1.0, "negative": -1.0},
                "cost": {},
            },
            "2026-07-23T11:00:00Z": {
                "usage": {},
                "cost": {"credit": -2.0},
            },
        },
        now,
        owner,
    )

    assert store.frontier() == "2026-07-23T10:00:00Z"

    with store._connection() as connection:
        connection.execute("UPDATE hourly SET total = 0")
        connection.execute("PRAGMA user_version = 1")
    reopened = cost_dashboard.UsageStore(database)
    assert reopened.frontier() == "2026-07-23T10:00:00Z"


def test_expired_refresh_owner_cannot_commit_over_new_lease(tmp_path):
    store = cost_dashboard.UsageStore(tmp_path / "usage.sqlite3")
    started = datetime(2026, 7, 23, 12, tzinfo=timezone.utc)
    status, stale_owner = store.acquire_refresh_lease(started)
    assert status == "acquired"
    replacement_time = (
        started + cost_dashboard.REFRESH_LEASE + timedelta(seconds=1)
    )
    status, current_owner = store.acquire_refresh_lease(replacement_time)
    assert status == "acquired"

    with pytest.raises(RuntimeError, match="ownership was lost"):
        store.commit_hourly_refresh(
            "2026-07-23T11:00:00Z",
            "2026-07-23T12:00:00Z",
            {"2026-07-23T11:00:00Z": _payload(9.0)},
            replacement_time,
            stale_owner,
        )
    assert (
        store.hourly_range("2026-07-23T11:00:00Z", "2026-07-23T12:00:00Z")
        == {}
    )

    store.commit_hourly_refresh(
        "2026-07-23T11:00:00Z",
        "2026-07-23T12:00:00Z",
        {"2026-07-23T11:00:00Z": _payload(1.0)},
        replacement_time,
        current_owner,
    )
    assert store.frontier() == "2026-07-23T11:00:00Z"


def test_persisted_lease_coalesces_and_rate_limits_force(tmp_path):
    store = cost_dashboard.UsageStore(tmp_path / "usage.sqlite3")
    now = datetime(2026, 7, 23, 12, tzinfo=timezone.utc)

    status, owner = store.acquire_refresh_lease(now, force=True)
    assert status == "acquired"
    assert store.metadata("last_fetch") is None
    assert store.metadata("last_force_attempt") == now.isoformat()
    assert store.acquire_refresh_lease(now)[0] == "in_progress"
    store.commit_hourly_refresh(
        "2026-07-23T11:00:00Z",
        "2026-07-23T12:00:00Z",
        {"2026-07-23T11:00:00Z": _payload(1.0, 1.0)},
        now,
        owner,
        force=True,
    )

    assert (
        store.acquire_refresh_lease(now + timedelta(minutes=1), force=True)[0]
        == "rate_limited"
    )
    status, owner = store.acquire_refresh_lease(now + timedelta(minutes=1))
    assert status == "acquired"
    store.release_refresh_lease(owner)

    status, owner = store.acquire_refresh_lease(
        now + cost_dashboard.REFRESH_LEASE + timedelta(seconds=1)
    )
    assert status == "acquired"
    store.release_refresh_lease(owner)


def test_coverage_reports_missing_hourly_history():
    expected = cost_dashboard._expected_buckets(
        "2026-07-01", "2026-07-01", "HOURLY"
    )
    shown = [(key, _payload()) for key, _ in expected[12:]]

    coverage = cost_dashboard._coverage(
        expected, shown, "HOURLY", "2026-07-01T23:00:00Z"
    )

    assert not coverage["complete"]
    assert coverage["requested_buckets"] == 24
    assert coverage["available_buckets"] == 12
    assert coverage["missing_buckets"] == 12
    assert "not available" in coverage["warning"]


def test_coverage_marks_latest_daily_hourly_bucket_partial():
    expected = cost_dashboard._expected_buckets(
        "2026-07-22", "2026-07-23", "DAILY"
    )
    shown = [(key, _payload()) for key, _ in expected]

    coverage = cost_dashboard._coverage(
        expected,
        shown,
        "DAILY",
        "2026-07-23T12:00:00Z",
        today=date(2026, 7, 23),
    )

    assert not coverage["complete"]
    assert coverage["missing_buckets"] == 0
    assert "2026-07-23 is partial through" in coverage["warning"]


def test_historical_daily_coverage_ignores_newer_hourly_high_water():
    expected = cost_dashboard._expected_buckets(
        "2026-07-01", "2026-07-02", "DAILY"
    )
    shown = [(key, _payload()) for key, _ in expected]

    coverage = cost_dashboard._coverage(
        expected,
        shown,
        "DAILY",
        "2026-07-23T12:00:00Z",
        today=date(2026, 7, 23),
    )

    assert coverage["complete"]
    assert coverage["warning"] is None


def test_local_server_enforces_token_origin_method_and_security_headers():
    server = ThreadingHTTPServer(("127.0.0.1", 0), cost_dashboard.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1", server.server_address[1]
    )
    try:
        connection.request("GET", "/")
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        assert response.status == 200
        assert cost_dashboard.API_TOKEN in body
        assert (
            "sha384-Mx5lkUEQPM1pOJCwFtUICyX45KNojXbkWdYhkKUKsbv391"
            "mavbfoAmONbzkgYPzR"
        ) in body
        assert response.getheader("Cache-Control") == "no-store, max-age=0"
        assert "default-src 'none'" in response.getheader(
            "Content-Security-Policy"
        )
        assert response.getheader("X-Frame-Options") == "DENY"
        assert (
            response.getheader("Cross-Origin-Embedder-Policy")
            == "require-corp"
        )

        connection.request("GET", "/api/runs")
        response = connection.getresponse()
        assert response.status == 403
        assert json.loads(response.read())["error"] == "forbidden-token"

        connection.request("GET", "/", headers={"Host": "attacker.example"})
        response = connection.getresponse()
        assert response.status == 403
        assert json.loads(response.read())["error"] == "forbidden-host"

        headers = {
            cost_dashboard.TOKEN_HEADER: cost_dashboard.API_TOKEN,
            "Origin": "https://attacker.example",
        }
        connection.request("GET", "/api/runs", headers=headers)
        response = connection.getresponse()
        assert response.status == 403
        assert json.loads(response.read())["error"] == "forbidden-origin"

        headers = {
            cost_dashboard.TOKEN_HEADER: cost_dashboard.API_TOKEN,
        }
        connection.request(
            "GET",
            (
                "/api/account?start=2026-07-01&end=2026-07-01"
                "&gran=DAILY&force=1"
            ),
            headers=headers,
        )
        response = connection.getresponse()
        assert response.status == 405
        assert json.loads(response.read())["error"] == "method-not-allowed"

        connection.request(
            "POST",
            (
                "/api/account/refresh?start=2026-07-01"
                "&end=2026-07-01&gran=DAILY"
            ),
            headers=headers,
        )
        response = connection.getresponse()
        assert response.status == 403
        assert json.loads(response.read())["error"] == "origin-required"
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join()
