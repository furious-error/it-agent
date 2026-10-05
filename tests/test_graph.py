"""LangGraph workflow with a scripted model: no network calls."""

from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.checkpoint import open_checkpointer
from app.graph import build_graph, interrupt_payload, waiting_for_approval
from app.mcp_client import InProcessToolClient
from app.messages import user_text
from app.state import initial_state
from mcp_server.tools import ToolKit


class ScriptedReasoner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []
        self.tools = []

    def generate(self, contents, tools=None):
        self.seen.append(contents)
        self.tools.append(tools)
        if not self.responses:
            raise AssertionError("scripted reasoner ran out of responses")
        return self.responses.pop(0)


class RecordingSummarizer:
    def __init__(self):
        self.calls = []

    def summarize(self, transcript: str) -> str:
        self.calls.append(transcript)
        return "orders-api degraded after a bad deployment"


def response_with(*parts):
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=list(parts)))])


def call_part(name, args, call_id="call-1"):
    return SimpleNamespace(text=None, thought=False, thought_signature=b"sig", function_call=SimpleNamespace(name=name, args=args, id=call_id))


def text_part(text):
    return SimpleNamespace(text=text, thought=False, thought_signature=None, function_call=None)


@pytest.fixture
def tk():
    toolkit = ToolKit()
    yield toolkit
    toolkit.store.close()


def _run(reasoner, toolkit, state, summarizer=None, checkpointer=None, thread="run"):
    summarizer = summarizer or RecordingSummarizer()
    checkpointer = checkpointer or InMemorySaver()
    client = InProcessToolClient(toolkit)
    graph = build_graph(reasoner=reasoner, tools=client, summarizer=summarizer, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 80}
    result = graph.invoke(state, config)
    return graph, config, result, client, summarizer


def test_read_only_investigation_uses_discovered_tools_and_records_the_audit(tk):
    reasoner = ScriptedReasoner([
        response_with(call_part("get_service_health", {"service": "orders-api"})),
        response_with(text_part("orders-api is degraded.")),
    ])
    _, _, result, client, summarizer = _run(reasoner, tk, initial_state(user_text("Investigate."), incident_id="INC-001", max_steps=6))
    assert result["stop_reason"] == "final"
    assert result["final_response"] == "orders-api is degraded."
    assert result["diagnosis"] == "orders-api is degraded."
    assert result["service"] == "orders-api"
    assert result["step_count"] == 2
    assert client.calls == [("get_service_health", {"service": "orders-api"})]
    assert result["audit"][0]["outcome"] == "success" and result["audit"][0]["executed"] is True
    assert {tool["name"] for tool in reasoner.tools[0]} == {tool["name"] for tool in tk.catalog()}
    assert summarizer.calls == []
    assert interrupt_payload(result) is None


def test_tool_error_is_returned_and_the_model_can_correct_it(tk):
    reasoner = ScriptedReasoner([
        response_with(call_part("search_logs", {"service": "notification-service", "start_time": "this morning"}, "bad")),
        response_with(call_part("search_logs", {"service": "notification-service", "start_time": "2026-09-30T09:30:00Z", "end_time": "2026-09-30T09:45:00Z", "level": "WARN"}, "good")),
        response_with(text_part("The backlog drained by 09:43. No ongoing fault.")),
    ])
    _, _, result, _, _ = _run(reasoner, tk, initial_state(user_text("Were notifications delayed?"), incident_id="INC-006", max_steps=6))
    assert [item["outcome"] for item in result["audit"]] == ["failure", "success"]
    assert result["audit"][0]["status"] == 400
    assert result["stop_reason"] == "final"


def test_high_risk_call_pauses_until_approval_then_executes(tk):
    reasoner = ScriptedReasoner([
        response_with(call_part("rollback_deployment", {"deployment_id": "deploy-8472", "reason": "bad db port"})),
        response_with(text_part("Rolled back deploy-8472. orders-api is healthy.")),
    ])
    graph, config, paused, client, _ = _run(reasoner, tk, initial_state(user_text("Fix orders-api."), incident_id="INC-001", max_steps=6))
    payload = interrupt_payload(paused)
    assert payload["actions"][0]["name"] == "rollback_deployment"
    assert payload["actions"][0]["arguments"]["deployment_id"] == "deploy-8472"
    assert "approval" in payload["actions"][0]["reason"]
    assert client.calls == []
    assert tk.get_service_health("orders-api")["status"] == "degraded"
    assert paused["remediation"].startswith("rollback_deployment")

    finished = graph.invoke(Command(resume="approve"), config)
    assert finished["stop_reason"] == "final"
    assert finished["audit"][-1]["tool"] == "rollback_deployment"
    assert finished["audit"][-1]["outcome"] == "success"
    assert tk.get_service_health("orders-api")["status"] == "healthy"
    assert tk.get_service_health("orders-api")["version"] == "v2.8.0"


def test_rejection_does_not_execute_and_the_same_action_is_not_asked_again(tk):
    reasoner = ScriptedReasoner([
        response_with(call_part("restart_service", {"service": "orders-api"}, "first")),
        response_with(call_part("restart_service", {"service": "orders-api"}, "again")),
        response_with(text_part("Restart was rejected. The fault is in the deployed version.")),
    ])
    graph, config, paused, client, _ = _run(reasoner, tk, initial_state(user_text("Fix it."), incident_id="INC-001", max_steps=6))
    assert interrupt_payload(paused)["actions"][0]["name"] == "restart_service"

    finished = graph.invoke(Command(resume="reject"), config)
    assert client.calls == []
    assert tk.get_service_health("orders-api")["status"] == "degraded"
    assert finished["stop_reason"] == "final"
    assert [item["outcome"] for item in finished["audit"]] == ["denied", "denied"]
    assert "already rejected" in finished["audit"][1]["arguments"] or finished["audit"][1]["status"] == 400
    # The second proposal was refused by the hook, so the graph did not pause again.
    assert interrupt_payload(finished) is None


def test_delete_is_blocked_by_the_hook_before_any_call(tk):
    reasoner = ScriptedReasoner([
        response_with(call_part("delete_resource", {"name": "orders-db"})),
        response_with(text_part("Deletion is not permitted.")),
    ])
    _, _, result, client, _ = _run(reasoner, tk, initial_state(user_text("Delete the database."), incident_id=None, max_steps=4))
    assert client.calls == []
    assert result["audit"][0]["outcome"] == "denied" and result["audit"][0]["status"] == 403
    assert interrupt_payload(result) is None


def test_integer_arguments_are_coerced_before_the_approved_call(tk):
    reasoner = ScriptedReasoner([
        response_with(call_part("scale_service", {"service": "api-gateway", "replicas": 8.0})),
        response_with(text_part("Scaled the gateway to 8 replicas.")),
    ])
    graph, config, paused, _, _ = _run(reasoner, tk, initial_state(user_text("Add capacity."), incident_id=None, max_steps=4))
    assert paused["pending_tool_call"]["arguments"]["replicas"] == 8
    assert "2x" not in interrupt_payload(paused)["actions"][0]["reason"]
    finished = graph.invoke(Command(resume="approve"), config)
    assert finished["audit"][0]["outcome"] == "success"
    assert tk.get_service_health("api-gateway")["replicas"] == 8


def test_compaction_runs_once_the_transcript_passes_ten_messages(tk):
    summarizer = RecordingSummarizer()
    reasoner = ScriptedReasoner([response_with(text_part("Continuing from the summary."))])
    state = initial_state(user_text("ignored"), incident_id="INC-001", max_steps=4)
    state["messages"] = [user_text(f"message {i}") for i in range(11)]
    _, _, result, _, _ = _run(reasoner, tk, state, summarizer=summarizer)
    assert summarizer.calls and "message 0" in summarizer.calls[0]
    assert result["summary"] == "orders-api degraded after a bad deployment"
    assert result["messages"][0]["parts"][0]["text"].startswith("Summary of the investigation so far:")
    assert result["stop_reason"] == "final"
    # 11 messages, 8 summarized into 1, plus the model's answer.
    assert len(result["messages"]) == 11 - 8 + 1 + 1


def test_max_steps_stops_without_another_model_call(tk):
    reasoner = ScriptedReasoner([
        response_with(call_part("get_incident", {"incident_id": "INC-004"})),
        response_with(text_part("should not be requested")),
    ])
    _, _, result, client, _ = _run(reasoner, tk, initial_state(user_text("Look."), incident_id="INC-004", max_steps=1))
    assert result["stop_reason"] == "max_steps"
    assert result["final_response"] is None
    assert client.calls == [("get_incident", {"incident_id": "INC-004"})]
    assert reasoner.responses  # the text turn was never consumed


def test_checkpoint_survives_a_new_process(tmp_path, tk):
    reasoner = ScriptedReasoner([
        response_with(call_part("rollback_deployment", {"deployment_id": "deploy-8472", "reason": "bad port"})),
        response_with(text_part("Rollback was rejected. Investigating further.")),
    ])
    path = tmp_path / "checkpoints.sqlite"
    connection, saver = open_checkpointer(path)
    client = InProcessToolClient(tk)
    graph = build_graph(reasoner=reasoner, tools=client, summarizer=RecordingSummarizer(), checkpointer=saver)
    config = {"configurable": {"thread_id": "thread-1"}, "recursion_limit": 80}
    paused = graph.invoke(initial_state(user_text("Fix orders."), incident_id="INC-001", max_steps=6), config)
    assert waiting_for_approval(graph, config, paused)["actions"][0]["name"] == "rollback_deployment"
    connection.close()

    # A restarted process opens the same file and compiles a new graph.
    connection2, saver2 = open_checkpointer(path)
    graph2 = build_graph(reasoner=reasoner, tools=client, summarizer=RecordingSummarizer(), checkpointer=saver2)
    assert waiting_for_approval(graph2, config, {})["incident_id"] == "INC-001"
    finished = graph2.invoke(Command(resume="reject"), config)
    connection2.close()
    assert finished["stop_reason"] == "final"
    assert "rejected" in finished["final_response"]
    assert tk.get_service_health("orders-api")["status"] == "degraded"
    assert client.calls == []
