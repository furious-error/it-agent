"""Langfuse tracing for observable agent events.

Import this module only after ``.env`` has been loaded. The SDK reads
LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL / LANGFUSE_HOST
at client construction.

Traces record model I/O, tokens, tool calls, hook decisions, and approval
events. They do not record hidden chain-of-thought or thought signatures.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from mcp_server.tools import READ_ONLY_TOOLS

_initialized = False


def tracing_enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def init_tracing() -> None:
    """Create the Langfuse client after credentials are in the environment."""
    global _initialized
    if _initialized or not tracing_enabled():
        return
    host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
    if host and "LANGFUSE_HOST" not in os.environ:
        os.environ["LANGFUSE_HOST"] = host
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "development")
    os.environ.setdefault("OTEL_SERVICE_NAME", "acmecloud-it-agent")
    from langfuse import get_client

    get_client()
    _initialized = True


def flush() -> None:
    if not tracing_enabled():
        return
    from langfuse import get_client

    get_client().flush()


def observation_type_for_tool(name: str) -> str:
    if name in READ_ONLY_TOOLS:
        return "retriever"
    return "tool"


@contextmanager
def observation(as_type: str, name: str, **kwargs: Any) -> Iterator[Any]:
    """Typed Langfuse observation, or a no-op when tracing is off."""
    if not tracing_enabled():
        yield _Noop()
        return
    from langfuse import get_client

    client = get_client()
    with client.start_as_current_observation(as_type=as_type, name=name, **kwargs) as obs:
        yield obs


@contextmanager
def investigation_trace(
    *,
    name: str,
    user_message: str | None,
    session_id: str,
    incident_id: str | None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """One graph invoke: an ``agent`` observation grouped into the thread session."""
    tags = ["incident-response"]
    if incident_id:
        tags.append(incident_id)
    extra = {"thread_id": session_id}
    if incident_id:
        extra["incident_id"] = incident_id
    if metadata:
        extra.update(metadata)
    if not tracing_enabled():
        yield _Noop()
        return
    from langfuse import get_client, propagate_attributes

    client = get_client()
    with client.start_as_current_observation(
        as_type="agent",
        name=name,
        input={"user_message": user_message} if user_message is not None else None,
    ) as root:
        with propagate_attributes(session_id=session_id, tags=tags, metadata=extra):
            yield root


def generation_io(contents: list[Any], response: Any) -> tuple[Any, Any, dict[str, int] | None]:
    """Compact generation I/O plus token usage. Thought signatures are dropped."""
    return _contents_for_trace(contents), _response_for_trace(response), _usage(response)


def _contents_for_trace(contents: list[Any]) -> list[dict[str, Any]]:
    rendered = []
    for item in contents:
        if isinstance(item, dict):
            rendered.append(_strip_signatures(item))
            continue
        role = getattr(item, "role", None) or "user"
        parts = []
        for part in getattr(item, "parts", None) or []:
            text = getattr(part, "text", None)
            if text and not getattr(part, "thought", False):
                parts.append({"type": "text", "text": text})
            call = getattr(part, "function_call", None)
            if call is not None and getattr(call, "name", None):
                parts.append({"type": "function_call", "name": call.name, "args": dict(getattr(call, "args", None) or {})})
            response = getattr(part, "function_response", None)
            if response is not None:
                parts.append({"type": "function_response", "name": getattr(response, "name", None), "response": _shorten(getattr(response, "response", None))})
        rendered.append({"role": role, "parts": parts})
    return rendered


def _response_for_trace(response: Any) -> dict[str, Any]:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return {"empty": True}
    content = getattr(candidates[0], "content", None)
    calls = []
    texts = []
    for part in getattr(content, "parts", None) or []:
        call = getattr(part, "function_call", None)
        if call is not None and getattr(call, "name", None):
            calls.append({"name": call.name, "args": dict(getattr(call, "args", None) or {})})
        text = getattr(part, "text", None)
        if text and not getattr(part, "thought", False):
            texts.append(text)
    out: dict[str, Any] = {}
    if calls:
        out["function_calls"] = calls
    if texts:
        out["text"] = "\n".join(texts)
    return out or {"empty": True}


def _usage(response: Any) -> dict[str, int] | None:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return None
    details = {}
    prompt = getattr(meta, "prompt_token_count", None)
    output = getattr(meta, "candidates_token_count", None)
    if prompt is not None:
        details["input"] = int(prompt)
    if output is not None:
        details["output"] = int(output)
    return details or None


def _strip_signatures(message: dict[str, Any]) -> dict[str, Any]:
    parts = []
    for part in message.get("parts") or []:
        if not isinstance(part, dict):
            continue
        cleaned = {k: v for k, v in part.items() if k != "thought_signature"}
        if "function_response" in cleaned:
            fr = dict(cleaned["function_response"])
            fr["response"] = _shorten(fr.get("response"))
            cleaned["function_response"] = fr
        parts.append(cleaned)
    return {"role": message.get("role"), "parts": parts}


def _shorten(value: Any, limit: int = 4000) -> Any:
    text = str(value)
    if len(text) <= limit:
        return value
    if isinstance(value, dict):
        return {**value, "_truncated": True}
    return text[:limit] + "..."


class _Noop:
    def update(self, **kwargs: Any) -> None:
        return None


# Names are stable on purpose: evaluators and dashboards filter on them.
PROPOSE_NEXT_ACTION = "propose-next-action"
SUMMARIZE_CONTEXT = "summarize-context"
LIST_TOOLS = "list-tools"
CHECK_TOOL_POLICY = "check-tool-policy"
AWAIT_HUMAN_APPROVAL = "await-human-approval"
RECORD_HUMAN_DECISION = "record-human-decision"
INVESTIGATE_INCIDENT = "investigate-incident"
RESUME_AFTER_APPROVAL = "resume-after-approval"
EVALUATE_ERROR_RECOVERY = "evaluate-error-recovery"
