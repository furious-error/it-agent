from app.telemetry import (
    INVESTIGATE_INCIDENT,
    PROPOSE_NEXT_ACTION,
    generation_io,
    observation,
    observation_type_for_tool,
    tracing_enabled,
)


def test_tracing_is_off_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert tracing_enabled() is False
    with observation(as_type="generation", name=PROPOSE_NEXT_ACTION) as obs:
        obs.update(output="ignored")


def test_read_only_tools_are_retrievers_and_actions_are_tools():
    assert observation_type_for_tool("search_logs") == "retriever"
    assert observation_type_for_tool("get_service_health") == "retriever"
    assert observation_type_for_tool("rollback_deployment") == "tool"
    assert observation_type_for_tool("add_incident_note") == "tool"


def test_generation_io_drops_thought_signatures_and_reads_usage():
    from types import SimpleNamespace

    contents = [SimpleNamespace(role="user", parts=[SimpleNamespace(text="hello", thought=False, function_call=None, function_response=None)])]
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="ok", thought=False, function_call=None),
                        SimpleNamespace(text="hidden", thought=True, function_call=None),
                    ]
                )
            )
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=11, candidates_token_count=3),
    )
    prompt, output, usage = generation_io(contents, response)
    assert prompt[0]["parts"][0]["text"] == "hello"
    assert output["text"] == "ok"
    assert usage == {"input": 11, "output": 3}
    assert INVESTIGATE_INCIDENT == "investigate-incident"
