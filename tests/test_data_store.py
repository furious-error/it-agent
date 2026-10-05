"""DataStore: loading, querying, mutation, reset, persistence."""

import pytest

from mcp_server.data_store import DataStore, normalize_time, parse_time


def test_loads_all_entities(store):
    assert len(store.list_services()) == 10
    assert len(store.list_incidents(limit=100)) == 10
    assert len(store.list_deployments(limit=100)) == 12
    assert len(store.list_databases()) == 7
    assert len(store.list_alerts()) == 9
    assert store.search_logs(limit=500)[1] == 196


def test_simulated_clock_from_meta(store):
    assert store.now_str() == "2026-09-30T10:20:00Z"
    store.advance_clock(90)
    assert store.now_str() == "2026-09-30T10:21:30Z"


def test_get_service_and_save(store):
    svc = store.get_service("orders-api")
    assert svc["status"] == "degraded" and svc["healthy_replicas"] == 2
    svc["status"] = "healthy"
    store.save_service(svc)
    assert store.get_service("orders-api")["status"] == "healthy"
    assert store.get_service("nope") is None


def test_list_incidents_filters(store):
    assert {i["incident_id"] for i in store.list_incidents(status="resolved")} == {"INC-007", "INC-008"}
    assert [i["incident_id"] for i in store.list_incidents(service="orders-api")] == ["INC-001", "INC-007"]
    assert store.next_incident_id() == "INC-011"


def test_list_deployments_order_and_filters(store):
    recent = store.list_deployments(limit=3)
    assert [d["deployment_id"] for d in recent] == ["deploy-8472", "deploy-8501", "deploy-8490"]
    orders = store.list_deployments(service="orders-api", limit=10)
    assert [d["deployment_id"] for d in orders] == ["deploy-8472", "deploy-8465", "deploy-8461", "deploy-8460"]
    since = store.list_deployments(since="2026-09-30T09:50:00Z", limit=10)
    assert {d["deployment_id"] for d in since} == {"deploy-8472", "deploy-8501", "deploy-8490", "deploy-8497"}
    assert store.next_deployment_id() == "deploy-8502"


def test_search_logs_filters_and_truncation(store):
    logs, total = store.search_logs(service="orders-api", level="ERROR", start_time="2026-09-30T10:00:00Z", end_time="2026-09-30T10:20:00Z", limit=3)
    assert total > 3 and len(logs) == 3
    assert [l["timestamp"] for l in logs] == sorted(l["timestamp"] for l in logs)
    assert logs[-1]["timestamp"] == "2026-09-30T10:17:09Z"  # most recent matches are kept

    logs, total = store.search_logs(keyword="JWT_SIGNING_KEY_V2")
    assert total == 1 and logs[0]["service"] == "auth-service"

    logs, _ = store.search_logs(keyword="connection REFUSED", service="orders-api")  # case-insensitive
    assert logs and all("refused" in l["message"].lower() for l in logs)


def test_metrics_range_and_latest(store):
    points = store.get_metrics("orders-api", start_time="2026-09-30T10:00:00Z", end_time="2026-09-30T10:05:00Z")
    assert [p["timestamp"][11:16] for p in points] == ["10:00", "10:01", "10:02", "10:03", "10:04", "10:05"]
    assert points[0]["error_rate"] < points[-1]["error_rate"]
    assert store.latest_metric("orders-api")["timestamp"] == "2026-09-30T10:20:00Z"
    assert store.latest_metric("redis-cache")["timestamp"] == "2026-09-30T09:50:00Z"
    assert store.latest_metric("unknown") is None


def test_dependencies_both_directions(store):
    deps = {d["name"] for d in store.get_dependencies("users-api")}
    assert deps == {"users-db", "redis-cache"}
    assert store.get_dependents("redis-cache") == ["users-api"]
    assert "api-gateway" in store.get_dependents("orders-api")


def test_audit_events_round_trip(store):
    ev = store.record_audit_event("tester", "restart_service", "orders-api", "success", {"reason": "x"})
    assert ev["event_id"] == 1
    events = store.list_audit_events()
    assert events[0]["details"] == {"reason": "x"} and events[0]["timestamp"] == "2026-09-30T10:20:00Z"


def test_reset_restores_seed(store):
    svc = store.get_service("orders-api")
    svc["status"] = "healthy"
    store.save_service(svc)
    store.advance_clock(600)
    store.record_audit_event("t", "a", None, "success")
    store.reset()
    assert store.get_service("orders-api")["status"] == "degraded"
    assert store.now_str() == "2026-09-30T10:20:00Z"
    assert store.list_audit_events() == []


def test_file_backed_store_persists(tmp_path):
    db = tmp_path / "incident.db"
    s1 = DataStore(db_path=str(db))
    svc = s1.get_service("redis-cache")
    svc["status"] = "healthy"
    s1.save_service(svc)
    s1.close()
    s2 = DataStore(db_path=str(db))  # must not re-seed over existing data
    assert s2.get_service("redis-cache")["status"] == "healthy"
    assert len(s2.list_services()) == 10
    s2.close()


def test_missing_seed_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DataStore(data_dir=tmp_path)


@pytest.mark.parametrize("raw,expected", [
    ("2026-09-30T10:00:00Z", "2026-09-30T10:00:00Z"),
    ("2026-09-30T15:30:00+05:30", "2026-09-30T10:00:00Z"),
    ("2026-09-30 10:00:00", "2026-09-30T10:00:00Z"),
])
def test_normalize_time(raw, expected):
    assert normalize_time(raw) == expected


@pytest.mark.parametrize("bad", ["invalid", "", "10 minutes ago", "2026-13-01T00:00:00Z", None])
def test_parse_time_rejects_garbage(bad):
    with pytest.raises((ValueError, TypeError)):
        parse_time(bad)
