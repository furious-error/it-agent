"""Deterministic guardrails around tool execution.

The model proposes a call. These functions decide whether it may run, whether a
human must approve it, or whether it is refused. They do not call the model.
"""

from __future__ import annotations

import json
from typing import Any

from mcp_server.tools import HIGH_RISK_TOOLS

BLOCKED_TOOLS = {"delete_resource", "drop_database", "delete_service"}


def call_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    return tool_name + ":" + json.dumps(arguments, sort_keys=True, default=str)


def pre_tool_hook(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    rejected_calls: list[str] | None = None,
    current_replicas: int | None = None,
) -> dict[str, Any]:
    """Return allowed, requires_approval, an HTTP-like status, and an error string.

    High-risk production actions are not executed here. The graph interrupts and
    only continues when the resume value approves this exact call.
    """
    rejected_calls = rejected_calls or []
    signature = call_signature(tool_name, arguments)
    if tool_name in BLOCKED_TOOLS or tool_name.startswith("delete"):
        return _decision(
            allowed=False,
            requires_approval=False,
            status=403,
            error=f"Refused: '{tool_name}' is blocked. Deleting resources is not permitted.",
            signature=signature,
        )
    if tool_name in HIGH_RISK_TOOLS:
        if signature in rejected_calls:
            return _decision(
                allowed=False,
                requires_approval=False,
                status=400,
                error=(
                    f"Human already rejected {tool_name} with these arguments. "
                    "Find another safe diagnostic or remediation approach. Do not ask for the same action again."
                ),
                signature=signature,
            )
        reason = "Production change requires human approval."
        if tool_name == "scale_service" and current_replicas and isinstance(arguments.get("replicas"), int):
            if arguments["replicas"] > current_replicas * 2:
                reason = (
                    f"Scaling {arguments.get('service', 'the service')} from {current_replicas} to "
                    f"{arguments['replicas']} replicas is more than 2x and requires human approval."
                )
        return _decision(allowed=False, requires_approval=True, status=400, error=reason, signature=signature)
    return _decision(allowed=True, requires_approval=False, status=200, error=None, signature=signature)


def post_tool_hook(tool_name: str, arguments: dict[str, Any], result: dict[str, Any], latency_ms: float, *, executed: bool) -> dict[str, Any]:
    """Audit record for one tool attempt. ``executed`` is false when a hook refused the call."""
    status = result.get("status")
    if not executed:
        outcome = "denied"
    elif status == 200:
        outcome = "success"
    else:
        outcome = "failure"
    return {
        "tool": tool_name,
        "arguments": arguments,
        "status": status,
        "outcome": outcome,
        "latency_ms": round(latency_ms, 2),
        "executed": executed,
    }


def rejection_result(tool_name: str) -> dict[str, Any]:
    return {
        "status": 400,
        "error": (
            f"Human rejected the proposed {tool_name}. "
            "Find another safe diagnostic or remediation approach."
        ),
    }


def _decision(*, allowed: bool, requires_approval: bool, status: int, error: str | None, signature: str) -> dict[str, Any]:
    return {
        "allowed": allowed,
        "requires_approval": requires_approval,
        "status": status,
        "error": error,
        "signature": signature,
    }
