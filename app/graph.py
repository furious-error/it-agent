"""LangGraph investigation workflow.

    discover → compact → reason → guard → approve? → execute → compact → reason → … → END

``discover`` learns the tool list from the client (MCP, or an in-process stand-in).
``guard`` applies the pre-hook. High-risk calls pause in ``approve`` via ``interrupt``
until a human resumes the same checkpoint. ``compact`` runs when the transcript
grows past ten messages.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .agent import ToolStep
from .compaction import COMPACTION_THRESHOLD, compact_messages
from .hooks import post_tool_hook, pre_tool_hook, rejection_result
from .messages import (
    calls_from_message,
    coerce_arguments,
    content_from_dict,
    content_to_dict,
    function_response_message,
    visible_text,
)
from .state import AgentState
from .telemetry import (
    AWAIT_HUMAN_APPROVAL,
    CHECK_TOOL_POLICY,
    LIST_TOOLS,
    RECORD_HUMAN_DECISION,
    observation,
    observation_type_for_tool,
)


class ToolClient(Protocol):
    def list_tools(self) -> list[dict[str, Any]]:
        """Discovered tools: name, description, input_schema."""

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute one tool. Return the status envelope."""


class Reasoner(Protocol):
    def generate(self, contents: list[Any], tools: list[dict[str, Any]] | None = None) -> Any:
        """One model turn."""


def build_graph(
    *,
    reasoner: Reasoner,
    tools: ToolClient,
    summarizer: Any,
    checkpointer: Any,
    on_step: Callable[[ToolStep], None] | None = None,
):
    """Compile the investigation graph. A checkpointer is required so interrupts can resume."""

    def discover(state: AgentState) -> dict[str, Any]:
        if state.get("available_tools"):
            return {}
        with observation(as_type="tool", name=LIST_TOOLS, input={}) as obs:
            catalog = tools.list_tools()
            obs.update(output={"count": len(catalog), "tools": [_described(item) for item in catalog]})
            return {"available_tools": catalog}

    def compact(state: AgentState) -> dict[str, Any]:
        summary, messages = compact_messages(
            state.get("messages") or [],
            summarizer.summarize,
            existing_summary=state.get("summary"),
        )
        if messages is state.get("messages"):
            return {}
        return {"summary": summary, "messages": messages}

    def reason(state: AgentState) -> dict[str, Any]:
        if state.get("step_count", 0) >= state.get("max_steps", 12):
            return {
                "stop_reason": "max_steps",
                "final_response": None,
                "pending_tool_call": None,
                "pending_tool_calls": [],
            }
        contents = [content_from_dict(message) for message in state.get("messages") or []]
        response = reasoner.generate(contents, tools=state.get("available_tools") or [])
        content = _model_content(response)
        step_count = state.get("step_count", 0) + 1
        if content is None:
            return {"stop_reason": "no_candidate", "final_response": None, "step_count": step_count, "pending_tool_calls": []}
        message = content_to_dict(content)
        messages = [*(state.get("messages") or []), message]
        calls = [_with_schema(call, state.get("available_tools") or []) for call in calls_from_message(message)]
        if not calls:
            text = visible_text(message) or None
            return {
                "messages": messages,
                "pending_tool_call": None,
                "pending_tool_calls": [],
                "final_response": text,
                "diagnosis": text,
                "stop_reason": "final",
                "step_count": step_count,
            }
        return {
            "messages": messages,
            "pending_tool_call": calls[0],
            "pending_tool_calls": calls,
            "final_response": None,
            "stop_reason": None,
            "approval_status": None,
            "approval_required": False,
            "step_count": step_count,
        }

    def guard(state: AgentState) -> dict[str, Any]:
        calls = state.get("pending_tool_calls") or []
        replicas = _current_replicas(tools, calls)
        decisions = [
            pre_tool_hook(
                call["name"],
                call["arguments"],
                rejected_calls=state.get("rejected_calls") or [],
                current_replicas=replicas if call["name"] == "scale_service" else None,
            )
            for call in calls
        ]
        needs_approval = any(decision["requires_approval"] for decision in decisions)
        remediation = state.get("remediation")
        if needs_approval:
            proposed = [call for call, decision in zip(calls, decisions) if decision["requires_approval"]]
            remediation = "; ".join(f"{call['name']} {call['arguments']}" for call in proposed)
        with observation(
            as_type="guardrail",
            name=CHECK_TOOL_POLICY,
            input=[{"name": call["name"], "arguments": call["arguments"]} for call in calls],
        ) as obs:
            obs.update(
                output={
                    "decisions": [
                        {"allowed": d["allowed"], "requires_approval": d["requires_approval"], "status": d["status"], "error": d.get("error")}
                        for d in decisions
                    ]
                }
            )
        return {
            "tool_decisions": decisions,
            "approval_required": needs_approval,
            "approval_status": "pending" if needs_approval else None,
            "remediation": remediation,
            "pending_tool_call": _approval_call(calls, decisions) or state.get("pending_tool_call"),
        }

    def approve(state: AgentState) -> dict[str, Any]:
        calls = state.get("pending_tool_calls") or []
        decisions = state.get("tool_decisions") or []
        actions = [
            {"name": call["name"], "arguments": call["arguments"], "reason": decision.get("error")}
            for call, decision in zip(calls, decisions)
            if decision.get("requires_approval")
        ]
        payload = {
            "actions": actions,
            "service": state.get("service"),
            "incident_id": state.get("incident_id"),
            "prompt": "Approve? [approve/reject]",
        }
        with observation(as_type="span", name=AWAIT_HUMAN_APPROVAL, input=payload) as waiting:
            waiting.update(output={"status": "waiting"})
        # Runs again from the top of this node on resume. Keep it free of side effects before this call.
        decision = interrupt(payload)
        approved = _is_approval(decision)
        with observation(as_type="span", name=RECORD_HUMAN_DECISION, input=payload) as recorded:
            recorded.update(output={"decision": "approved" if approved else "rejected"})
        return {"approval_status": "approved" if approved else "rejected", "approval_required": False}

    def execute(state: AgentState) -> dict[str, Any]:
        calls = state.get("pending_tool_calls") or []
        decisions = state.get("tool_decisions") or []
        approved = state.get("approval_status") == "approved"
        rejected = list(state.get("rejected_calls") or [])
        audit = list(state.get("audit") or [])
        responses: list[dict[str, Any]] = []
        last_result: dict[str, Any] | None = None
        service = state.get("service")
        environment = state.get("environment")
        for call, decision in zip(calls, decisions):
            # A high-risk call is allowed=False until this resume approves it.
            blocked = decision.get("requires_approval") and not approved
            if blocked:
                result = rejection_result(call["name"])
                rejected.append(decision["signature"])
                executed = False
                latency_ms = 0.0
                _record_tool_observation(call, result, executed=False, reason="human_rejected", description=_tool_description(state, call["name"]))
            elif not decision.get("allowed") and not decision.get("requires_approval"):
                result = {"status": decision.get("status", 400), "error": decision.get("error")}
                executed = False
                latency_ms = 0.0
                _record_tool_observation(call, result, executed=False, reason="hook_denied", description=_tool_description(state, call["name"]))
            else:
                started = time.perf_counter()
                description = _tool_description(state, call["name"])
                with observation(
                    as_type=observation_type_for_tool(call["name"]),
                    name=call["name"],
                    input=call["arguments"],
                    metadata={"mcp": True, "description": description},
                ) as tool_obs:
                    try:
                        result = tools.call_tool(call["name"], call["arguments"])
                    except Exception as exc:  # the tool process failed; the model can adapt
                        result = {"status": 500, "error": f"{type(exc).__name__}: {exc}"}
                    tool_obs.update(output=result, metadata={"mcp": True, "description": description, "status": result.get("status"), "executed": True})
                latency_ms = (time.perf_counter() - started) * 1000
                executed = True
                service, environment = _learn_context(call["name"], result, service, environment)
            audit.append(post_tool_hook(call["name"], call["arguments"], result, latency_ms, executed=executed))
            if on_step:
                on_step(ToolStep(name=call["name"], arguments=call["arguments"], response=result, risk=_risk(decision)))
            responses.append({"name": call["name"], "id": call.get("id"), "response": result})
            last_result = result
        messages = [*(state.get("messages") or []), function_response_message(responses)]
        return {
            "messages": messages,
            "last_tool_result": last_result,
            "audit": audit,
            "rejected_calls": rejected,
            "service": service,
            "environment": environment,
            "pending_tool_call": None,
            "pending_tool_calls": [],
            "tool_decisions": [],
            "approval_required": False,
            "approval_status": None,
        }

    graph = StateGraph(AgentState)
    graph.add_node("discover", discover)
    graph.add_node("compact", compact)
    graph.add_node("reason", reason)
    graph.add_node("guard", guard)
    graph.add_node("approve", approve)
    graph.add_node("execute", execute)
    graph.add_edge(START, "discover")
    graph.add_edge("discover", "compact")
    graph.add_edge("compact", "reason")
    graph.add_conditional_edges("reason", _after_reason, {"guard": "guard", END: END})
    graph.add_conditional_edges("guard", _after_guard, {"approve": "approve", "execute": "execute"})
    graph.add_edge("approve", "execute")
    graph.add_edge("execute", "compact")
    return graph.compile(checkpointer=checkpointer)


def interrupt_payload(result: Any) -> dict[str, Any] | None:
    """The value passed to ``interrupt``, if this invoke paused for approval."""
    raw = result.get("__interrupt__") if isinstance(result, dict) else None
    if not raw:
        return None
    first = raw[0]
    value = first.value if hasattr(first, "value") else first
    return value if isinstance(value, dict) else {"value": value}


def waiting_for_approval(graph: Any, config: dict[str, Any], result: Any) -> dict[str, Any] | None:
    payload = interrupt_payload(result)
    if payload:
        return payload
    snapshot = graph.get_state(config)
    for item in getattr(snapshot, "interrupts", None) or ():
        value = item.value if hasattr(item, "value") else item
        if isinstance(value, dict):
            return value
    return None


def _after_reason(state: AgentState) -> str:
    if state.get("stop_reason") or not state.get("pending_tool_calls"):
        return END
    return "guard"


def _after_guard(state: AgentState) -> str:
    if state.get("approval_required"):
        return "approve"
    return "execute"


def _model_content(response: Any) -> Any | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    return getattr(candidates[0], "content", None)


def _with_schema(call: dict[str, Any], available: list[dict[str, Any]]) -> dict[str, Any]:
    schema = next((tool.get("input_schema") for tool in available if tool.get("name") == call["name"]), None)
    return {**call, "arguments": coerce_arguments(call["arguments"], schema)}


def _approval_call(calls: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> dict[str, Any] | None:
    for call, decision in zip(calls, decisions):
        if decision.get("requires_approval"):
            return call
    return calls[0] if calls else None


def _current_replicas(tools: ToolClient, calls: list[dict[str, Any]]) -> int | None:
    scale = next((call for call in calls if call["name"] == "scale_service"), None)
    service = (scale or {}).get("arguments", {}).get("service") if scale else None
    if not service:
        return None
    try:
        health = tools.call_tool("get_service_health", {"service": service})
    except Exception:
        return None
    if health.get("status") != 200:
        return None
    replicas = (health.get("result") or {}).get("replicas")
    return replicas if isinstance(replicas, int) else None


def _learn_context(name: str, result: dict[str, Any], service: str | None, environment: str | None) -> tuple[str | None, str | None]:
    if result.get("status") != 200 or not isinstance(result.get("result"), dict):
        return service, environment
    body = result["result"]
    if name == "get_incident":
        return body.get("service") or service, body.get("environment") or environment
    if name == "get_service_health":
        return body.get("service") or service, environment
    return service, environment


def _is_approval(decision: Any) -> bool:
    return str(decision).strip().lower() in {"approve", "approved", "yes", "y"}


def _described(tool: dict[str, Any]) -> dict[str, str]:
    return {"name": tool.get("name") or "", "description": tool.get("description") or ""}


def _tool_description(state: AgentState, name: str) -> str:
    for tool in state.get("available_tools") or []:
        if tool.get("name") == name:
            return str(tool.get("description") or "")
    return ""


def _record_tool_observation(call: dict[str, Any], result: dict[str, Any], *, executed: bool, reason: str, description: str) -> None:
    with observation(
        as_type=observation_type_for_tool(call["name"]),
        name=call["name"],
        input=call["arguments"],
        metadata={"mcp": False, "executed": executed, "reason": reason, "description": description},
    ) as tool_obs:
        tool_obs.update(output=result)


def _risk(decision: dict[str, Any]) -> str:
    if decision.get("requires_approval") or decision.get("status") == 403:
        return "high"
    return "read_only"


# Re-exported so callers can explain the threshold without importing compaction internals.
__all__ = ["COMPACTION_THRESHOLD", "build_graph", "interrupt_payload"]
