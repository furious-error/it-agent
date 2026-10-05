from app.hooks import call_signature, post_tool_hook, pre_tool_hook, rejection_result


def test_read_only_and_low_risk_tools_are_allowed():
    for name, args in (("search_logs", {"service": "orders-api"}), ("add_incident_note", {"incident_id": "INC-001", "note": "checking"})):
        decision = pre_tool_hook(name, args)
        assert decision["allowed"] is True
        assert decision["requires_approval"] is False


def test_high_risk_tools_require_approval():
    for name, args in (
        ("restart_service", {"service": "orders-api"}),
        ("rollback_deployment", {"deployment_id": "deploy-8472"}),
        ("disable_feature", {"service": "inventory-api", "feature_flag": "async_stock_sync"}),
    ):
        decision = pre_tool_hook(name, args)
        assert decision["allowed"] is False
        assert decision["requires_approval"] is True
        assert decision["status"] == 400
        assert "approval" in decision["error"]


def test_scaling_past_double_explains_why_approval_is_required():
    modest = pre_tool_hook("scale_service", {"service": "api-gateway", "replicas": 8}, current_replicas=6)
    assert modest["requires_approval"] is True
    assert "2x" not in modest["error"]
    doubled = pre_tool_hook("scale_service", {"service": "api-gateway", "replicas": 13}, current_replicas=6)
    assert doubled["requires_approval"] is True
    assert "more than 2x" in doubled["error"]


def test_delete_is_refused_without_asking():
    decision = pre_tool_hook("delete_resource", {"name": "orders-db"})
    assert decision["allowed"] is False
    assert decision["requires_approval"] is False
    assert decision["status"] == 403


def test_a_rejected_action_is_not_asked_again():
    signature = call_signature("restart_service", {"service": "payments-api"})
    decision = pre_tool_hook("restart_service", {"service": "payments-api"}, rejected_calls=[signature])
    assert decision["requires_approval"] is False
    assert decision["allowed"] is False
    assert "already rejected" in decision["error"]


def test_post_hook_records_latency_and_outcome():
    success = post_tool_hook("get_service_health", {"service": "orders-api"}, {"status": 200, "result": {}}, 12.345, executed=True)
    assert success["outcome"] == "success" and success["latency_ms"] == 12.35 and success["executed"] is True
    failure = post_tool_hook("search_logs", {}, {"status": 400, "error": "bad"}, 3, executed=True)
    assert failure["outcome"] == "failure"
    denied = post_tool_hook("delete_resource", {}, {"status": 403, "error": "no"}, 0, executed=False)
    assert denied["outcome"] == "denied" and denied["executed"] is False


def test_rejection_result_tells_the_model_to_change_approach():
    result = rejection_result("rollback_deployment")
    assert result["status"] == 400
    assert "rejected the proposed rollback_deployment" in result["error"]
    assert "another" in result["error"]
