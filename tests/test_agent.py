"""The tool loop, with a scripted model. No network calls."""

from types import SimpleNamespace

import pytest
from google.genai import types

from app.agent import GeminiReasoner, IncidentAgent, _visible_text
from app.config import Settings
from app.prompts import RULES, SYSTEM_PROMPT, incident_user_message, opening_message
from app.schemas import function_declarations
from mcp_server.tools import ToolKit


class ScriptedReasoner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []

    def generate(self, contents):
        self.seen.append(contents)
        if not self.responses:
            raise AssertionError("scripted reasoner ran out of responses")
        return self.responses.pop(0)


def response_with(*parts, text=None):
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=list(parts)))], text=text)


def call_part(name, args, call_id="call-1", signature=b"sig"):
    return SimpleNamespace(text=None, thought=False, thought_signature=signature, function_call=SimpleNamespace(name=name, args=args, id=call_id))


def text_part(text, *, thought=False):
    return SimpleNamespace(text=text, thought=thought, thought_signature=None, function_call=None)


@pytest.fixture
def tk():
    toolkit = ToolKit()
    yield toolkit
    toolkit.store.close()


def test_declarations_cover_the_catalog_and_mark_risk(tk):
    catalog = tk.catalog()
    declarations = {item.name: item for item in function_declarations(catalog)}
    assert set(declarations) == {tool["name"] for tool in catalog}
    assert declarations["rollback_deployment"].description.startswith("[high]")
    assert declarations["search_logs"].description.startswith("[read_only]")
    assert declarations["add_incident_note"].description.startswith("[low]")
    schema = declarations["search_logs"].parameters_json_schema
    assert schema["properties"]["level"]["enum"] == ["DEBUG", "ERROR", "INFO", "WARN"]
    assert "service" in declarations["get_service_health"].parameters_json_schema["required"]
    assert "required" not in schema or "level" not in schema.get("required", [])


def test_system_prompt_contains_the_operating_rules():
    for index, rule in enumerate(RULES, start=1):
        assert f"{index}. {rule}" in SYSTEM_PROMPT
    assert "restart_service, rollback_deployment, scale_service and disable_feature" in SYSTEM_PROMPT


def test_opening_message_adds_clock_and_incident_only_for_the_first_turn():
    text = opening_message("Orders are failing.", now="2026-09-30T10:20:00Z", incident_id="INC-001")
    assert "Orders are failing." in text
    assert "2026-09-30T10:20:00Z" in text
    assert "get_incident" in text
    incident = ToolKit().get_incident("INC-001")
    report = incident_user_message(incident)
    assert "HTTP 500" in report
    assert "without my approval" in report


def test_loop_executes_the_proposed_call_and_returns_the_result(tk):
    reasoner = ScriptedReasoner([
        response_with(call_part("get_service_health", {"service": "orders-api"}, call_id="c1", signature=b"keep-me")),
        response_with(text_part("orders-api is degraded: 2 of 4 replicas, error rate above 30%.", thought=False), text="orders-api is degraded"),
    ])
    agent = IncidentAgent(reasoner=reasoner, toolkit=tk, max_steps=4)
    result = agent.send("Investigate orders-api.")

    assert result.stop_reason == "final"
    assert result.model_turns == 2
    assert result.final_response.startswith("orders-api is degraded")
    assert result.steps[0].succeeded
    assert result.steps[0].response["result"]["status"] == "degraded"

    # The model content is kept verbatim, signature included.
    assert agent.contents[1].parts[0].thought_signature == b"keep-me"
    function_response = agent.contents[2].parts[0].function_response
    assert function_response.name == "get_service_health"
    assert function_response.id == "c1"
    assert function_response.response["status"] == 200
    # The second model turn saw the tool result.
    assert reasoner.seen[1][-1].parts[0].function_response.response["result"]["service"] == "orders-api"


def test_tool_error_is_returned_to_the_model_which_can_correct_the_call(tk):
    reasoner = ScriptedReasoner([
        response_with(call_part("search_logs", {"service": "notification-service", "start_time": "this morning"}, call_id="bad")),
        response_with(call_part("search_logs", {"service": "notification-service", "start_time": "2026-09-30T09:30:00Z", "end_time": "2026-09-30T09:45:00Z", "level": "WARN"}, call_id="good")),
        response_with(text_part("The email provider throttled briefly; the backlog drained by 09:43.")),
    ])
    agent = IncidentAgent(reasoner=reasoner, toolkit=tk)
    result = agent.send("Were notifications delayed?")

    assert [step.response["status"] for step in result.steps] == [400, 200]
    assert "Invalid time range" in result.steps[0].response["error"]
    assert any("429" in line["message"] for line in result.steps[1].response["result"]["logs"])
    assert result.stop_reason == "final"
    # The corrected call was produced after the model was shown the 400.
    assert reasoner.seen[1][-1].parts[0].function_response.response["status"] == 400


def test_parallel_calls_in_one_turn_are_all_executed(tk):
    reasoner = ScriptedReasoner([
        response_with(
            call_part("get_service_health", {"service": "orders-api"}, call_id="a"),
            call_part("get_recent_deployments", {"service": "orders-api", "limit": 1}, call_id="b"),
        ),
        response_with(text_part("Health is degraded and deploy-8472 is the newest release.")),
    ])
    result = IncidentAgent(reasoner=reasoner, toolkit=tk).send("Check orders.")
    assert [step.name for step in result.steps] == ["get_service_health", "get_recent_deployments"]
    assert result.steps[1].response["result"]["deployments"][0]["deployment_id"] == "deploy-8472"
    assert result.model_turns == 2


def test_integer_arguments_sent_as_floats_are_coerced(tk):
    reasoner = ScriptedReasoner([
        response_with(call_part("scale_service", {"service": "api-gateway", "replicas": 8.0, "reason": "capacity"})),
        response_with(text_part("Scaled.")),
    ])
    result = IncidentAgent(reasoner=reasoner, toolkit=tk).send("Scale the gateway.")
    assert result.steps[0].arguments["replicas"] == 8
    assert result.steps[0].succeeded
    assert tk.get_service_health("api-gateway")["replicas"] == 8


def test_high_risk_call_is_executed_when_the_model_requests_it(tk):
    """Milestone 2 has no approval gate. The prompt tells the model to ask first; the loop still runs the call."""
    reasoner = ScriptedReasoner([
        response_with(call_part("rollback_deployment", {"deployment_id": "deploy-8472", "reason": "bad db port"})),
        response_with(text_part("Rolled back deploy-8472. orders-api is healthy.")),
    ])
    seen = []
    result = IncidentAgent(reasoner=reasoner, toolkit=tk, on_step=seen.append).send("Approved: roll back deploy-8472.")
    assert result.steps[0].risk == "high" and result.steps[0].succeeded
    assert result.steps[0].response["result"]["health_improved"] is True
    assert tk.get_service_health("orders-api")["status"] == "healthy"
    assert seen[0].name == "rollback_deployment"


def test_unknown_tool_error_is_fed_back(tk):
    reasoner = ScriptedReasoner([
        response_with(call_part("drop_database", {"name": "orders-db"})),
        response_with(text_part("That tool does not exist.")),
    ])
    result = IncidentAgent(reasoner=reasoner, toolkit=tk).send("Drop it.")
    assert result.steps[0].response["status"] == 404
    assert result.stop_reason == "final"


def test_max_steps_stops_after_a_tool_call_with_no_follow_up(tk):
    reasoner = ScriptedReasoner([
        response_with(call_part("get_incident", {"incident_id": "INC-004"})),
        response_with(text_part("should not be requested")),
    ])
    result = IncidentAgent(reasoner=reasoner, toolkit=tk, max_steps=1).send("Look at INC-004.")
    assert result.stop_reason == "max_steps"
    assert result.final_response is None
    assert result.steps[0].succeeded
    assert reasoner.responses  # the text turn was never consumed


def test_thought_text_is_not_the_final_answer():
    content = SimpleNamespace(parts=[text_part("hidden", thought=True), text_part("visible")])
    assert _visible_text(content) == "visible"


def test_empty_candidate_stops(tk):
    reasoner = ScriptedReasoner([SimpleNamespace(candidates=[])])
    result = IncidentAgent(reasoner=reasoner, toolkit=tk).send("Hello.")
    assert result.stop_reason == "no_candidate"
    assert result.final_response is None


def test_follow_up_message_continues_the_same_conversation(tk):
    reasoner = ScriptedReasoner([
        response_with(text_part("I need approval to roll back deploy-8472.")),
        response_with(call_part("rollback_deployment", {"deployment_id": "deploy-8472"})),
        response_with(text_part("Rollback completed and orders-api is healthy.")),
    ])
    agent = IncidentAgent(reasoner=reasoner, toolkit=tk)
    first = agent.send("Investigate.")
    assert first.stop_reason == "final" and first.steps == []
    second = agent.send("Approved: rollback_deployment deploy-8472.")
    assert second.steps[0].succeeded
    assert tk.get_service_health("orders-api")["status"] == "healthy"
    roles = [getattr(item, "role", None) for item in agent.contents]
    assert roles == ["user", "model", "user", "model", "user", "model"]


class RecordingClient:
    def __init__(self):
        self.models = self
        self.kwargs = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return response_with(text_part("ok"))


def test_reasoner_disables_automatic_execution_and_passes_schemas(tk):
    client = RecordingClient()
    settings = Settings(api_key="test-key", model="gemini-3.5-flash", thinking="low", temperature=0.2)
    reasoner = GeminiReasoner(client, settings, tk)
    reasoner.generate(["contents-go-here"])

    sent = client.kwargs
    assert sent["model"] == "gemini-3.5-flash"
    assert sent["contents"] == ["contents-go-here"]
    assert sent["config"].automatic_function_calling.disable is True
    assert sent["config"].temperature == 0.2
    assert sent["config"].thinking_config.include_thoughts is False
    names = [decl.name for decl in sent["config"].tools[0].function_declarations]
    assert len(names) == 17
    assert "get_service_health" in names
    assert "SRE incident response agent" in sent["config"].system_instruction
