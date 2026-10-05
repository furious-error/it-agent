"""The real MCP server, over stdio, in front of the simulated infrastructure."""

from app.mcp_client import McpToolClient


def test_server_lists_tools_and_returns_structured_results(tmp_path):
    client = McpToolClient(tmp_path / "infra.db")
    try:
        tools = client.list_tools()
        names = {tool["name"] for tool in tools}
        assert "get_service_health" in names
        assert "rollback_deployment" in names
        assert "search_logs" in names
        assert len(names) == 17
        health_tool = next(tool for tool in tools if tool["name"] == "get_service_health")
        assert "service" in health_tool["input_schema"]["properties"]
        assert health_tool["description"].startswith("[read_only]")

        health = client.call_tool("get_service_health", {"service": "orders-api"})
        assert health["status"] == 200
        assert health["result"]["status"] == "degraded"
        assert health["result"]["healthy_replicas"] == 2

        bad = client.call_tool("search_logs", {"service": "orders-api", "start_time": "10 minutes ago"})
        assert bad["status"] == 400
        assert "Invalid time range" in bad["error"]
    finally:
        client.close()


def test_server_keeps_mutations_in_its_database_file(tmp_path):
    db_path = tmp_path / "infra.db"
    first = McpToolClient(db_path)
    try:
        note = first.call_tool("add_incident_note", {"incident_id": "INC-001", "note": "seen from MCP"})
        assert note["status"] == 200
    finally:
        first.close()

    second = McpToolClient(db_path)
    try:
        incident = second.call_tool("get_incident", {"incident_id": "INC-001"})
        assert incident["status"] == 200
        assert incident["result"]["notes"][-1]["note"] == "seen from MCP"
    finally:
        second.close()
