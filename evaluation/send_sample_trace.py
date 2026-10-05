"""Send one scripted investigation to Langfuse so the tree can be audited.

    python -m evaluation.send_sample_trace

Uses the in-process tool client and a scripted model. No Gemini call.
Requires LANGFUSE_* in .env.
"""

from __future__ import annotations

from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver

from app.config import load_dotenv
from app.graph import build_graph
from app.mcp_client import InProcessToolClient
from app.messages import user_text
from app.state import initial_state
from app.telemetry import (
    INVESTIGATE_INCIDENT,
    PROPOSE_NEXT_ACTION,
    flush,
    generation_io,
    init_tracing,
    investigation_trace,
    observation,
    tracing_enabled,
)
from mcp_server.tools import ToolKit


def response_with(*parts):
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=list(parts)))])


def call_part(name, args, call_id="c"):
    return SimpleNamespace(text=None, thought=False, thought_signature=None, function_call=SimpleNamespace(name=name, args=args, id=call_id))


def text_part(text):
    return SimpleNamespace(text=text, thought=False, thought_signature=None, function_call=None)


class Scripted:
    def __init__(self):
        self.responses = [
            response_with(call_part("search_logs", {"service": "notification-service", "start_time": "this morning"}, "bad")),
            response_with(call_part("search_logs", {"service": "notification-service", "start_time": "2026-09-30T09:30:00Z", "end_time": "2026-09-30T09:45:00Z", "level": "WARN"}, "good")),
            response_with(text_part("The email provider throttled briefly; the backlog drained by 09:43. No ongoing fault.")),
        ]

    def generate(self, contents, tools=None):
        with observation(as_type="generation", name=PROPOSE_NEXT_ACTION, model="scripted") as gen:
            response = self.responses.pop(0)
            prompt, output, _usage = generation_io(contents, response)
            gen.update(input=prompt, output=output, model="scripted")
            return response


class NoSummarizer:
    def summarize(self, transcript: str) -> str:
        return "unused"


def main() -> int:
    load_dotenv()
    if not tracing_enabled():
        print("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required.")
        return 1
    init_tracing()
    toolkit = ToolKit()
    graph = build_graph(
        reasoner=Scripted(),
        tools=InProcessToolClient(toolkit),
        summarizer=NoSummarizer(),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "sample-inc006"}, "recursion_limit": 40}
    message = "Customers report delayed email notifications. Investigate INC-006."
    with investigation_trace(
        name=INVESTIGATE_INCIDENT,
        user_message=message,
        session_id="sample-inc006",
        incident_id="INC-006",
        metadata={"source": "send_sample_trace"},
    ) as root:
        result = graph.invoke(initial_state(user_text(message), incident_id="INC-006", max_steps=6), config)
        root.update(output={"stop_reason": result.get("stop_reason"), "final_response": result.get("final_response")})
        from langfuse import get_client

        client = get_client()
        url = client.get_trace_url()
    flush()
    print("sent investigate-incident for INC-006 (session sample-inc006)")
    print(result.get("final_response"))
    if url:
        print(url)
    toolkit.store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
