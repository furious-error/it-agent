"""One investigation process: checkpoint, MCP world, and Langfuse flush.

The CLI and the HTTP API both use this. A run id is the checkpoint thread id
and the filename of ``<runs>/<id>.db``.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from google import genai
from langgraph.types import Command

from .agent import GeminiReasoner
from .checkpoint import open_checkpointer
from .compaction import GeminiSummarizer
from .config import Settings, load_settings
from .graph import build_graph, waiting_for_approval
from .mcp_client import McpToolClient
from .messages import user_text
from .prompts import incident_user_message, opening_message
from .reports import record_user_report
from .state import initial_state
from .telemetry import (
    INVESTIGATE_INCIDENT,
    RESUME_AFTER_APPROVAL,
    flush,
    init_tracing,
    investigation_trace,
)


def checkpoint_path() -> Path:
    return Path(os.environ.get("CHECKPOINT_PATH", "var/checkpoints.sqlite"))


def runs_dir() -> Path:
    return Path(os.environ.get("RUNS_DIR", "var/runs"))


def seq_file() -> Path:
    configured = os.environ.get("INCIDENT_SEQ_FILE")
    if configured:
        return Path(configured)
    return checkpoint_path().parent / "incident_seq.sqlite"


def share_incident_sequence() -> None:
    """Make MCP subprocesses allocate ids from the same counter as this process."""
    os.environ["INCIDENT_SEQ_FILE"] = str(seq_file())


def status_path(run_id: str) -> Path:
    return runs_dir() / f"{run_id}.status.json"


def read_status(run_id: str) -> dict[str, Any] | None:
    path = status_path(run_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_status(record: dict[str, Any]) -> None:
    path = status_path(record["run_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize_decision(value: str) -> str:
    answer = value.strip().lower()
    if answer in {"approve", "approved", "yes", "y"}:
        return "approve"
    if answer in {"reject", "rejected", "no", "n"}:
        return "reject"
    raise ValueError("decision must be approve or reject")


@contextmanager
def investigation_session(thread_id: str, settings: Settings | None = None) -> Iterator[tuple[Any, dict[str, Any], McpToolClient]]:
    """Open the checkpoint and the MCP server for one invoke, then close both."""
    settings = settings or load_settings()
    init_tracing()
    share_incident_sequence()
    connection, checkpointer = open_checkpointer(checkpoint_path())
    client = McpToolClient(runs_dir() / f"{thread_id}.db")
    try:
        graph = build_graph(
            reasoner=GeminiReasoner(genai.Client(api_key=settings.api_key), settings),
            tools=client,
            summarizer=GeminiSummarizer(genai.Client(api_key=settings.api_key), settings.summary_model or settings.model),
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": max(100, settings.max_steps * 8)}
        yield graph, config, client
    finally:
        flush()
        client.close()
        connection.close()


def begin_run(message: str | None, incident_id: str | None, run_id: str) -> dict[str, Any]:
    """Create the run database and resolve the incident id. Does not call the model."""
    share_incident_sequence()
    runs_dir().mkdir(parents=True, exist_ok=True)
    database = runs_dir() / f"{run_id}.db"
    client = McpToolClient(database)
    try:
        if incident_id:
            fetched = client.call_tool("get_incident", {"incident_id": incident_id})
            if fetched.get("status") != 200:
                raise LookupError(fetched.get("error") or f"Unknown incident '{incident_id}'.")
        else:
            if message is None or not message.strip():
                raise ValueError("message is empty")
            incident_id = record_user_report(database, message)
    finally:
        client.close()
    record = {
        "run_id": run_id,
        "incident_id": incident_id,
        "status": "running",
        "message": None,
        "final_response": None,
        "approval": None,
        "stop_reason": None,
        "error": None,
    }
    write_status(record)
    return record


def opening_for(client: McpToolClient, message: str | None, incident_id: str) -> str:
    if message is None or not message.strip():
        fetched = client.call_tool("get_incident", {"incident_id": incident_id})
        if fetched.get("status") != 200:
            raise LookupError(fetched.get("error") or f"Unknown incident '{incident_id}'.")
        message = incident_user_message(fetched["result"])
    now = "2026-09-30T10:20:00Z"
    meta = Path("data/meta.json")
    if meta.is_file():
        now = json.loads(meta.read_text(encoding="utf-8")).get("simulated_now", now)
    return opening_message(message, now=now, incident_id=incident_id)


def invoke_traced(graph: Any, pending: Any, config: dict[str, Any], thread_id: str, incident_id: str | None) -> dict[str, Any]:
    if isinstance(pending, Command):
        name = RESUME_AFTER_APPROVAL
        user_message = str(getattr(pending, "resume", ""))
    else:
        name = INVESTIGATE_INCIDENT
        user_message = _user_text(pending)
    with investigation_trace(name=name, user_message=user_message, session_id=thread_id, incident_id=incident_id) as root:
        result = graph.invoke(pending, config)
        root.update(
            output={
                "stop_reason": result.get("stop_reason"),
                "final_response": result.get("final_response"),
                "approval_status": result.get("approval_status"),
                "step_count": result.get("step_count"),
                "service": result.get("service"),
            }
        )
        return result


def settle_run(run_id: str, incident_id: str | None, graph: Any, config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    approval = waiting_for_approval(graph, config, result)
    if approval:
        record = _base(run_id, incident_id or result.get("incident_id"), "waiting_for_approval")
        record["approval"] = approval
        record["message"] = _approval_message(approval)
        write_status(record)
        return record
    stop = result.get("stop_reason") or "failed"
    record = _base(run_id, incident_id or result.get("incident_id"), "final" if stop == "final" else stop)
    record["stop_reason"] = stop
    record["final_response"] = result.get("final_response")
    record["message"] = result.get("final_response")
    write_status(record)
    return record


def fail_run(run_id: str, incident_id: str | None, exc: Exception) -> None:
    current = read_status(run_id) or _base(run_id, incident_id, "failed")
    current["status"] = "failed"
    current["error"] = f"{type(exc).__name__}: {exc}"
    current["message"] = current["error"]
    write_status(current)


def execute_investigation(run_id: str, message: str | None, incident_id: str) -> None:
    """Run the graph from the opening report until it finishes or pauses for approval."""
    try:
        settings = load_settings()
        with investigation_session(run_id, settings) as (graph, config, client):
            text = opening_for(client, message, incident_id)
            pending = initial_state(user_text(text), incident_id=incident_id, max_steps=settings.max_steps)
            result = invoke_traced(graph, pending, config, run_id, incident_id)
            settle_run(run_id, incident_id, graph, config, result)
    except Exception as exc:
        fail_run(run_id, incident_id, exc)


def resume_investigation(run_id: str, decision: str) -> None:
    current = read_status(run_id) or {}
    incident_id = current.get("incident_id")
    try:
        with investigation_session(run_id) as (graph, config, _client):
            result = invoke_traced(graph, Command(resume=decision), config, run_id, incident_id)
            settle_run(run_id, incident_id, graph, config, result)
    except Exception as exc:
        fail_run(run_id, incident_id, exc)


def _base(run_id: str, incident_id: str | None, status: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "incident_id": incident_id,
        "status": status,
        "message": None,
        "final_response": None,
        "approval": None,
        "stop_reason": None,
        "error": None,
    }


def _approval_message(payload: dict[str, Any]) -> str:
    actions = payload.get("actions") or []
    if not actions:
        return "A high-risk action requires approval."
    names = ", ".join(action.get("name") or "action" for action in actions)
    return f"{names} requires approval."


def _user_text(pending: Any) -> str | None:
    if not isinstance(pending, dict):
        return None
    messages = pending.get("messages") or []
    if not messages:
        return None
    parts = messages[-1].get("parts") or []
    texts = [part.get("text") for part in parts if isinstance(part, dict) and part.get("text")]
    return "\n".join(texts) or None
