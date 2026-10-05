"""Integrity checks on the JSON seed files: references resolve, timelines are consistent."""

import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "data"
EVAL = Path(__file__).resolve().parent.parent / "evaluation"


def load(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def seed():
    return {k: load(f"{k}.json") for k in ["services", "incidents", "deployments", "logs", "metrics", "databases", "alerts", "dependencies"]}


def test_all_seed_files_exist():
    for name in ["services", "incidents", "deployments", "logs", "metrics", "databases", "alerts", "dependencies", "meta"]:
        assert (DATA / f"{name}.json").exists(), name


def test_ten_incidents_with_unique_ids(seed):
    ids = [i["incident_id"] for i in seed["incidents"]]
    assert len(ids) == 10
    assert len(set(ids)) == 10


def test_incident_services_exist(seed):
    names = {s["name"] for s in seed["services"]}
    for inc in seed["incidents"]:
        assert inc["service"] in names, inc["incident_id"]


def test_deployment_services_exist_and_ids_unique(seed):
    names = {s["name"] for s in seed["services"]}
    ids = [d["deployment_id"] for d in seed["deployments"]]
    assert len(ids) == len(set(ids))
    for d in seed["deployments"]:
        assert d["service"] in names, d["deployment_id"]


def test_current_versions_match_latest_successful_deployment(seed):
    """A service's version must equal the version of its most recent completed (non in-progress) deployment."""
    for svc in seed["services"]:
        deps = [d for d in seed["deployments"] if d["service"] == svc["name"] and d["status"] in ("successful", "rolled_back")]
        if not deps:
            continue
        latest = max(deps, key=lambda d: d["deployed_at"])
        assert svc["version"] == latest["version"], svc["name"]


def test_alert_references_resolve(seed):
    names = {s["name"] for s in seed["services"]}
    incident_ids = {i["incident_id"] for i in seed["incidents"]}
    for a in seed["alerts"]:
        assert a["service"] in names, a["alert_id"]
        if a["incident_id"]:
            assert a["incident_id"] in incident_ids, a["alert_id"]
        if a["status"] == "resolved":
            assert a["resolved_at"]


def test_dependency_targets_resolve(seed):
    names = {s["name"] for s in seed["services"]} | {d["name"] for d in seed["databases"]}
    for record in seed["dependencies"]:
        assert record["service"] in {s["name"] for s in seed["services"]}
        for dep in record["depends_on"]:
            if dep["type"] in ("service", "database", "cache"):
                assert dep["name"] in names, dep


def test_logs_sorted_and_reference_known_services(seed):
    names = {s["name"] for s in seed["services"]}
    timestamps = [l["timestamp"] for l in seed["logs"]]
    assert timestamps == sorted(timestamps)
    assert all(l["service"] in names for l in seed["logs"])
    assert all(l["level"] in {"INFO", "WARN", "ERROR", "DEBUG"} for l in seed["logs"])
    assert len({l["log_id"] for l in seed["logs"]}) == len(seed["logs"])


def test_metrics_cover_every_service_each_minute_except_redis_outage(seed):
    per_service = {}
    for m in seed["metrics"]:
        per_service.setdefault(m["service"], []).append(m["timestamp"])
    for svc in seed["services"]:
        points = per_service[svc["name"]]
        assert len(points) == len(set(points)), svc["name"]
        if svc["name"] == "redis-cache":
            assert max(points) == "2026-09-30T09:50:00Z"  # exporter down from 09:51
        else:
            assert len(points) == 51, svc["name"]


def test_inc001_timeline_deployment_precedes_errors(seed):
    deploy = next(d for d in seed["deployments"] if d["deployment_id"] == "deploy-8472")
    inc = next(i for i in seed["incidents"] if i["incident_id"] == "INC-001")
    first_error = min(l["timestamp"] for l in seed["logs"] if l["service"] == "orders-api" and l["level"] == "ERROR" and l["timestamp"] >= "2026-09-30")
    assert deploy["deployed_at"] < first_error < inc["detected_at"]
    orders = sorted((m for m in seed["metrics"] if m["service"] == "orders-api"), key=lambda m: m["timestamp"])
    before = [m["error_rate"] for m in orders if m["timestamp"] < deploy["deployed_at"]]
    after = [m["error_rate"] for m in orders if m["timestamp"] >= "2026-09-30T10:05:00Z"]
    assert max(before) < 0.02
    assert min(after) > 0.3


def test_inc010_contradiction_logs_vs_db_metrics(seed):
    search_db = next(d for d in seed["databases"] if d["name"] == "search-db")
    assert search_db["status"] == "healthy"
    timeouts = [l for l in seed["logs"] if l["service"] == "search-api" and "connection timeout" in l["message"]]
    assert timeouts and all("pg-replica-old" in l["message"] for l in timeouts)


def test_ground_truth_covers_every_incident(seed):
    truth = json.loads((EVAL / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    assert {t["incident_id"] for t in truth} == {i["incident_id"] for i in seed["incidents"]}
    for t in truth:
        if t["expected_remediation"]["tool"] == "rollback_deployment":
            dep_id = t["expected_remediation"]["arguments"]["deployment_id"]
            assert any(d["deployment_id"] == dep_id for d in seed["deployments"]), t["incident_id"]


def test_hidden_simulation_blocks_are_present_in_seed_only(seed):
    assert all("simulation" in s for s in seed["services"])
    assert any("simulation" in d for d in seed["deployments"])  # the failing rollback
