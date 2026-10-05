"""Explicit state passed between LangGraph nodes.

Values are JSON-friendly. The transcript stores Gemini message parts, including
opaque thought signatures the API needs on the next turn. It does not store
hidden chain-of-thought text.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    incident_id: str | None
    service: str | None
    environment: str | None
    available_tools: list[dict[str, Any]]
    pending_tool_call: dict[str, Any] | None
    pending_tool_calls: list[dict[str, Any]]
    tool_decisions: list[dict[str, Any]]
    last_tool_result: dict[str, Any] | None
    approval_required: bool
    approval_status: str | None
    rejected_calls: list[str]
    diagnosis: str | None
    remediation: str | None
    step_count: int
    max_steps: int
    summary: str | None
    final_response: str | None
    audit: list[dict[str, Any]]
    stop_reason: str | None


def initial_state(user_message: dict[str, Any], *, incident_id: str | None, max_steps: int) -> AgentState:
    return {
        "messages": [user_message],
        "incident_id": incident_id,
        "service": None,
        "environment": None,
        "available_tools": [],
        "pending_tool_call": None,
        "pending_tool_calls": [],
        "tool_decisions": [],
        "last_tool_result": None,
        "approval_required": False,
        "approval_status": None,
        "rejected_calls": [],
        "diagnosis": None,
        "remediation": None,
        "step_count": 0,
        "max_steps": max_steps,
        "summary": None,
        "final_response": None,
        "audit": [],
        "stop_reason": None,
    }
