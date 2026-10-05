"""Tool layer: read-only diagnostics, incident actions, simulated remediations, router."""

import pytest

from mcp_server.tools import HIGH_RISK_TOOLS, LOW_RISK_TOOLS, READ_ONLY_TOOLS, ToolError, ToolKit, tool_risk


# --------------------------------------------------------------------------- #
# catalog / router
# --------------------------------------------------------------------------- #
def test_catalog_lists_all_tools_with_risk(tk):
    catalog = {t["name"]: t for t in tk.catalog()}
    assert set(catalog) == READ_ONLY_TOOLS | LOW_RISK_TOOLS | HIGH_RISK_TOOLS
    assert len(catalog) == 17
    assert catalog["rollback_deployment"]["risk"] == "high"
    assert catalog["search_logs"]["risk"] == "read_only"
    assert catalog["add_incident_note"]["risk"] == "low"
    assert all(t["description"] for t in catalog.values())
    assert {p["name"] for p in catalog["search_logs"]["parameters"]} == {"service", "level", "keyword", "start_time", "end_time", "limit"}
    assert tool_risk("made_up") == "unknown"


def test_execute_success_envelope(tk):
    out = tk.execute("get_service_health", {"service": "orders-api"})
    assert out["status"] == 200
    assert out["result"]["status"] == "degraded"


def test_execute_unknown_tool_and_bad_arguments(tk):
    assert tk.execute("delete_everything", {})["status"] == 404
    bad = tk.execute("get_service_health", {"svc": "orders-api"})
    assert bad["status"] == 400 and "Invalid arguments" in bad["error"]
    assert tk.execute("get_service_health", {})["status"] == 400
    assert tk.execute("get_service_health", "orders-api")["status"] == 400  # type: ignore[arg-type]


def test_execute_maps_tool_error_to_structured_error(tk):
    out = tk.execute("get_service_health", {"service": "ghost-api"})
    assert out == {"status": 404, "error": "Unknown service 'ghost-api'.", "details": {"known_services": tk.store.service_names()}}


# --------------------------------------------------------------------------- #
# read-only diagnostics
# --------------------------------------------------------------------------- #
def test_get_incident(tk):
    inc = tk.get_incident("INC-001")
    assert inc["service"] == "orders-api" and inc["severity"] == "SEV-1" and inc["root_cause"] is None
    with pytest.raises(ToolError) as e:
        tk.get_incident("INC-999")
    assert e.value.status == 404


def test_list_incidents(tk):
    out = tk.list_incidents(status="investigating")
    assert out["count"] == 7
    assert all(i["status"] == "investigating" for i in out["incidents"])
    with pytest.raises(ToolError):
        tk.list_incidents(status="bogus")


def test_get_service_health_merges_latest_metrics_and_hides_simulation(tk):
    h = tk.get_service_health("orders-api")
    assert h["replicas"] == 4 and h["healthy_replicas"] == 2
    assert h["error_rate"] > 0.3 and h["p95_latency_ms"] > 1500
    assert h["metrics_as_of"] == "2026-09-30T10:20:00Z"
    assert "simulation" not in h and "metrics_stale" not in h


def test_get_service_health_flags_stale_metrics_and_rollout(tk):
    redis = tk.get_service_health("redis-cache")
    assert redis["status"] == "down" and redis["metrics_stale"] is True
    auth = tk.get_service_health("auth-service")
    assert auth["rollout"]["status"] == "stuck" and auth["rollout"]["ready_replicas"] == 0


def test_get_recent_deployments_and_get_deployment(tk):
    recent = tk.get_recent_deployments(service="orders-api", limit=2)
    assert [d["deployment_id"] for d in recent["deployments"]] == ["deploy-8472", "deploy-8465"]
    assert "changes" not in recent["deployments"][0]
    full = tk.get_deployment("deploy-8472")
    assert any("6432" in c for c in full["changes"])
    assert "simulation" not in tk.get_deployment("deploy-8501")
    with pytest.raises(ToolError) as e:
        tk.get_recent_deployments(since="yesterday")
    assert e.value.status == 400


def test_search_logs_defaults_to_last_hour(tk):
    out = tk.search_logs(service="orders-api", level="ERROR")
    assert out["query"]["start_time"] == "2026-09-30T09:20:00Z"
    assert out["query"]["end_time"] == "2026-09-30T10:20:00Z"
    assert out["total_matches"] == out["returned"] > 5
    assert all(l["level"] == "ERROR" for l in out["logs"])
    assert any("6432" in l["message"] for l in out["logs"])


def test_search_logs_keyword_and_truncation(tk):
    out = tk.search_logs(keyword="health check", limit=5)
    assert out["returned"] == 5 and out["truncated"] is True and out["total_matches"] > 5


@pytest.mark.parametrize("kwargs,fragment", [
    ({"start_time": "invalid"}, "not a valid ISO 8601"),
    ({"start_time": "10 minutes ago"}, "not a valid ISO 8601"),
    ({"start_time": "2026-09-30T10:10:00Z", "end_time": "2026-09-30T10:00:00Z"}, "must be before"),
    ({"start_time": "2026-09-28T00:00:00Z", "end_time": "2026-09-30T10:00:00Z"}, "exceeds the maximum"),
    ({"level": "CRITICAL"}, "Unknown log level"),
    ({"limit": 0}, "'limit' must be"),
    ({"limit": 10_000}, "'limit' must be"),
    ({"service": "ghost-api"}, "Unknown service"),
])
def test_search_logs_structured_errors(tk, kwargs, fragment):
    """INC-006 relies on the agent receiving a structured 400 and correcting its request."""
    out = tk.execute("search_logs", kwargs)
    assert out["status"] in (400, 404)
    assert fragment in out["error"]


def test_search_logs_error_recovery_path(tk):
    bad = tk.execute("search_logs", {"service": "notification-service", "start_time": "this morning"})
    assert bad["status"] == 400 and "Invalid time range" in bad["error"]
    good = tk.execute("search_logs", {"service": "notification-service", "start_time": "2026-09-30T09:30:00Z", "end_time": "2026-09-30T09:45:00Z", "level": "WARN"})
    assert good["status"] == 200
    assert any("429" in l["message"] for l in good["result"]["logs"])


def test_get_metrics_summary_shows_ramp(tk):
    out = tk.get_metrics("orders-api", start_time="2026-09-30T09:55:00Z", end_time="2026-09-30T10:10:00Z")
    s = out["summary"]
    assert s["points"] == 16
    assert s["error_rate"]["first"] < 0.02 and s["error_rate"]["latest"] > 0.3
    assert s["p95_latency_ms"]["max"] > 1500


def test_get_metrics_empty_window_for_down_target(tk):
    out = tk.get_metrics("redis-cache")  # default window is the last 20 minutes; redis stopped at 09:50
    assert out["summary"]["points"] == 0 and "note" in out["summary"]


def test_get_metrics_errors(tk):
    with pytest.raises(ToolError) as e:
        tk.get_metrics("orders-api", start_time="nope")
    assert e.value.status == 400
    with pytest.raises(ToolError) as e:
        tk.get_metrics("ghost")
    assert e.value.status == 404


def test_get_database_status(tk):
    all_dbs = tk.get_database_status()
    assert all_dbs["count"] == 7
    payments = tk.get_database_status("payments-db")
    assert payments["connections_active"] == payments["connections_max"] == 200
    search = tk.get_database_status("search-db")
    assert search["status"] == "healthy" and search["host"] == "pg-replica-02.internal"
    with pytest.raises(ToolError) as e:
        tk.get_database_status("nope-db")
    assert e.value.status == 404


def test_get_alerts(tk):
    firing = tk.get_alerts(status="firing")
    assert firing["count"] == 7
    gw = tk.get_alerts(service="api-gateway")
    assert gw["alerts"][0]["status"] == "resolved"
    with pytest.raises(ToolError):
        tk.get_alerts(status="pending")


def test_get_dependencies_includes_status(tk):
    out = tk.get_dependencies("users-api")
    redis = next(d for d in out["depends_on"] if d["name"] == "redis-cache")
    assert redis["status"] == "down"
    assert {d["name"] for d in out["depended_on_by"]} == {"api-gateway"}
    ext = next(d for d in tk.get_dependencies("payments-api")["depends_on"] if d["type"] == "external")
    assert ext["status"] == "unknown"


# --------------------------------------------------------------------------- #
# low-risk actions
# --------------------------------------------------------------------------- #
def test_add_incident_note(tk):
    out = tk.add_incident_note("INC-001", "Errors correlate with deploy-8472.")
    assert out["note_count"] == 1 and out["note"]["author"] == "sre-agent"
    assert tk.get_incident("INC-001")["notes"][0]["note"].startswith("Errors correlate")
    assert tk.store.list_audit_events()[-1]["action"] == "add_incident_note"
    with pytest.raises(ToolError):
        tk.add_incident_note("INC-001", "   ")


def test_create_incident(tk):
    inc = tk.create_incident("Test incident", "api-gateway", "sev-3", "Just testing.")
    assert inc["incident_id"] == "INC-011" and inc["severity"] == "SEV-3" and inc["status"] == "open"
    assert tk.get_incident("INC-011")["environment"] == "production"
    with pytest.raises(ToolError):
        tk.create_incident("x", "api-gateway", "SEV-9", "y")


def test_update_incident_resolution_sets_resolved_at_and_timeline(tk):
    out = tk.update_incident("INC-004", status="resolved", root_cause="Transient traffic burst.", resolution="No action required.")
    assert out["resolved_at"] == "2026-09-30T10:20:00Z"
    assert out["timeline"][-1]["event"].startswith("Status changed investigating -> resolved")
    with pytest.raises(ToolError):
        tk.update_incident("INC-004")
    with pytest.raises(ToolError):
        tk.update_incident("INC-004", status="done")


# --------------------------------------------------------------------------- #
# high-risk actions: INC-001 bad deployment
# --------------------------------------------------------------------------- #
def test_restart_orders_api_does_not_fix_bad_deployment(tk):
    out = tk.restart_service("orders-api", reason="try restart")
    assert out["health_improved"] is False
    assert out["after"]["status"] == "degraded"
    assert tk.store.now_str() == "2026-09-30T10:21:00Z"


def test_rollback_deploy_8472_heals_orders_api(tk):
    out = tk.rollback_deployment("deploy-8472", reason="Deployment preceded 500s")
    assert out["health_improved"] is True
    assert out["version"] == "v2.8.0"
    assert out["rollback_deployment_id"] == "deploy-8502"

    health = tk.get_service_health("orders-api")
    assert health["status"] == "healthy" and health["healthy_replicas"] == 4
    assert health["error_rate"] <= 0.01  # fresh baseline metric sample at the new "now"
    assert health["metrics_as_of"] == "2026-09-30T10:21:00Z"

    assert tk.get_deployment("deploy-8472")["status"] == "rolled_back"
    newest = tk.get_recent_deployments(service="orders-api", limit=1)["deployments"][0]
    assert newest["type"] == "rollback" and newest["rollback_of"] == "deploy-8472"
    assert tk.get_alerts(service="orders-api", status="firing")["count"] == 0
    audit = tk.store.list_audit_events()[-1]
    assert audit["action"] == "rollback_deployment" and audit["outcome"] == "success"


def test_rollback_twice_is_conflict(tk):
    tk.rollback_deployment("deploy-8472")
    with pytest.raises(ToolError) as e:
        tk.rollback_deployment("deploy-8472")
    assert e.value.status == 409
    with pytest.raises(ToolError) as e:
        tk.rollback_deployment("deploy-8461")  # historical rollback deployment
    assert e.value.status == 409


def test_rollback_wrong_deployment_does_not_heal(tk):
    out = tk.rollback_deployment("deploy-8465")  # older orders-api release, not the cause
    assert out["health_improved"] is False
    assert tk.get_service_health("orders-api")["status"] == "degraded"


# --------------------------------------------------------------------------- #
# INC-002 payments pool exhaustion
# --------------------------------------------------------------------------- #
def test_payments_restart_clears_pool_and_heals(tk):
    out = tk.restart_service("payments-api")
    assert out["health_improved"] is True
    assert tk.get_database_status("payments-db")["status"] == "healthy"
    assert tk.get_database_status("payments-db")["connections_active"] < 200


def test_payments_disable_retry_wrapper_heals(tk):
    out = tk.disable_feature("payments-api", "provider_retry_wrapper")
    assert out["health_improved"] is True
    assert tk.get_service_health("payments-api")["feature_flags"]["provider_retry_wrapper"] is False


# --------------------------------------------------------------------------- #
# INC-003 redis down -> users-api
# --------------------------------------------------------------------------- #
def test_restarting_users_api_does_not_help_but_restarting_redis_does(tk):
    out = tk.restart_service("users-api")
    assert out["health_improved"] is False and "redis-cache" in out["note"]

    out = tk.restart_service("redis-cache")
    assert out["health_improved"] is True
    assert tk.get_service_health("redis-cache")["status"] == "healthy"
    assert tk.get_service_health("users-api")["status"] == "healthy"  # dependency propagation
    assert tk.get_database_status("redis-cache")["status"] == "healthy"
    assert tk.get_alerts(status="firing")["count"] == 5  # RedisDown + users-api HighErrorRate resolved


# --------------------------------------------------------------------------- #
# INC-005 stuck rollout
# --------------------------------------------------------------------------- #
def test_rollback_stuck_deployment_clears_rollout(tk):
    out = tk.rollback_deployment("deploy-8490")
    assert out["health_improved"] is True and out["version"] == "v3.1.4"
    health = tk.get_service_health("auth-service")
    assert health["status"] == "healthy" and "rollout" not in health
    assert tk.get_deployment("deploy-8490")["status"] == "rolled_back"


# --------------------------------------------------------------------------- #
# INC-009 rollback fails, feature flag fixes
# --------------------------------------------------------------------------- #
def test_inc009_rollback_fails_then_disable_feature_heals(tk):
    out = tk.execute("rollback_deployment", {"deployment_id": "deploy-8501", "reason": "bad release"})
    assert out["status"] == 500
    assert "not found" in out["error"] and out["details"]["previous_version"] == "v1.4.9"
    assert tk.get_service_health("inventory-api")["status"] == "degraded"
    assert tk.get_deployment("deploy-8501")["status"] == "successful"  # unchanged
    audit = tk.store.list_audit_events()[-1]
    assert audit["action"] == "rollback_deployment" and audit["outcome"] == "failure"
    errors = tk.search_logs(service="inventory-api", level="ERROR", keyword="Rollback failed")
    assert errors["total_matches"] == 1

    out = tk.execute("disable_feature", {"service": "inventory-api", "feature_flag": "async_stock_sync"})
    assert out["status"] == 200 and out["result"]["health_improved"] is True
    assert tk.get_service_health("inventory-api")["status"] == "healthy"


def test_disable_feature_errors(tk):
    with pytest.raises(ToolError) as e:
        tk.disable_feature("inventory-api", "nonexistent_flag")
    assert e.value.status == 404 and "async_stock_sync" in e.value.details["known_flags"]
    with pytest.raises(ToolError) as e:
        tk.disable_feature("inventory-api", "bulk_import")  # already off
    assert e.value.status == 409
    out = tk.disable_feature("api-gateway", "rate_limiting_v2")  # unrelated flag on healthy service
    assert out["health_improved"] is False


# --------------------------------------------------------------------------- #
# INC-010 contradictory signals
# --------------------------------------------------------------------------- #
def test_inc010_restart_postgres_is_useless_rollback_config_fixes(tk):
    assert tk.restart_service("postgres-db")["health_improved"] is False
    assert tk.get_service_health("search-api")["status"] == "degraded"
    out = tk.rollback_deployment("deploy-8497")
    assert out["health_improved"] is True and out["version"] == "v1.9.2"


# --------------------------------------------------------------------------- #
# scale_service
# --------------------------------------------------------------------------- #
def test_scale_service(tk):
    out = tk.scale_service("api-gateway", 8)
    assert out["replicas"] == 8 and out["healthy_replicas"] == 8 and out["note"] is None
    out = tk.scale_service("orders-api", 8)  # degraded: unhealthy ratio preserved
    assert out["healthy_replicas"] == 4 and "does not address" in out["note"]
    for bad in (0, 21, "4", 4.0, True):
        with pytest.raises(ToolError) as e:
            tk.scale_service("orders-api", bad)  # type: ignore[arg-type]
        assert e.value.status == 400
    with pytest.raises(ToolError) as e:
        tk.scale_service("orders-api", 8)
    assert e.value.status == 409


# --------------------------------------------------------------------------- #
# isolation
# --------------------------------------------------------------------------- #
def test_toolkits_on_separate_stores_are_isolated():
    a, b = ToolKit(), ToolKit()
    a.rollback_deployment("deploy-8472")
    assert a.get_service_health("orders-api")["status"] == "healthy"
    assert b.get_service_health("orders-api")["status"] == "degraded"
