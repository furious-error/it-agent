"""Command-line runner for one incident investigation.

    python -m app.main --incident INC-001
    python -m app.main --resume <thread-id>

The graph pauses before any high-risk tool. Approve or reject at the prompt.
The checkpoint and the simulated infrastructure file are both on disk, so a
later process can resume the same thread with --resume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from google import genai
from langgraph.types import Command

from .agent import GeminiReasoner, ToolStep
from .checkpoint import open_checkpointer
from .compaction import GeminiSummarizer
from .config import ConfigError, load_settings
from .graph import build_graph, waiting_for_approval
from .mcp_client import McpToolClient
from .messages import user_text
from .prompts import incident_user_message, opening_message
from .reports import record_user_report
from .runtime import invoke_traced, share_incident_sequence
from .state import initial_state
from .telemetry import (
    flush,
    init_tracing,
)

VAR = Path("var")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Investigate a simulated AcmeCloud incident with Gemini.")
    parser.add_argument("--incident", help="Seeded incident id, for example INC-001.")
    parser.add_argument("--message", help="User report. Defaults to the incident title and description.")
    parser.add_argument("--thread", help="Checkpoint thread id. A new id is generated when omitted.")
    parser.add_argument("--resume", action="store_true", help="Continue an existing thread instead of starting one.")
    parser.add_argument("--once", action="store_true", help="Exit at the first answer or approval pause instead of reading follow-up lines.")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum model turns for the run.")
    parser.add_argument("--checkpoint", type=Path, default=VAR / "checkpoints.sqlite", help="SQLite file for graph checkpoints.")
    parser.add_argument("--trace", type=Path, help="Write the final state as JSON to this path.")
    parser.add_argument("--verbose", action="store_true", help="Print full tool results.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.resume and not args.incident and not args.message:
        build_parser().error("provide --incident, --message, or --resume")
    try:
        settings = load_settings()
        init_tracing()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    if args.max_steps is not None:
        if args.max_steps < 1:
            print("Configuration error: --max-steps must be at least 1.", file=sys.stderr)
            return 1
        settings = type(settings)(
            api_key=settings.api_key,
            model=settings.model,
            max_steps=args.max_steps,
            temperature=settings.temperature,
            thinking=settings.thinking,
            summary_model=settings.summary_model,
        )

    thread_id = args.thread or (None if args.resume else uuid.uuid4().hex[:12])
    if args.resume and not thread_id:
        print("Configuration error: --resume requires --thread.", file=sys.stderr)
        return 1
    assert thread_id is not None
    os.environ.setdefault("INCIDENT_SEQ_FILE", str(args.checkpoint.parent / "incident_seq.sqlite"))
    share_incident_sequence()

    connection, checkpointer = open_checkpointer(args.checkpoint)
    infrastructure = VAR / "runs" / f"{thread_id}.db"
    client = McpToolClient(infrastructure)
    exit_code = 0
    try:
        graph = build_graph(
            reasoner=GeminiReasoner(genai.Client(api_key=settings.api_key), settings),
            tools=client,
            summarizer=GeminiSummarizer(genai.Client(api_key=settings.api_key), settings.summary_model or settings.model),
            checkpointer=checkpointer,
            on_step=lambda step: print_step(step, verbose=args.verbose),
        )
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": max(100, settings.max_steps * 8)}
        print(f"model: {settings.model}")
        print(f"thread: {thread_id}")
        print(f"checkpoint: {args.checkpoint}")
        print(f"infrastructure: {infrastructure}\n")

        if args.resume:
            snapshot = graph.get_state(config)
            if not snapshot.values:
                print(f"No checkpoint for thread {thread_id}.", file=sys.stderr)
                return 1
            approval = waiting_for_approval(graph, config, {})
            if approval:
                print(_format_approval(approval))
                if not sys.stdin.isatty():
                    print(f"\nPaused for approval. Resume with: python -m app.main --resume --thread {thread_id}")
                    return 3
                answer = _read_decision()
                if answer is None:
                    return 3
                pending = Command(resume=answer)
            else:
                _print_answer(snapshot.values)
                pending = _follow_up_input(graph, config, once=args.once)
                if pending is None:
                    return 0
        else:
            incident_id = args.incident
            if incident_id is None:
                incident_id = record_user_report(infrastructure, args.message)
            message = _opening(client, message=args.message, incident_id=incident_id)
            print(f"incident: {incident_id}")
            print(f"report: {message}\n")
            pending = initial_state(user_text(message), incident_id=incident_id, max_steps=settings.max_steps)
        if args.resume:
            incident_id = (graph.get_state(config).values or {}).get("incident_id") or args.incident

        interactive = sys.stdin.isatty() and not args.once
        result: dict[str, Any] = {}
        while pending is not None:
            try:
                result = invoke_traced(graph, pending, config, thread_id, incident_id)
            except Exception as exc:
                print(f"Run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
            approval = waiting_for_approval(graph, config, result)
            if approval:
                print(_format_approval(approval))
                if not interactive:
                    print(f"\nPaused for approval. Resume with: python -m app.main --resume --thread {thread_id}")
                    exit_code = 3
                    break
                answer = _read_decision()
                if answer is None:
                    exit_code = 3
                    break
                pending = Command(resume=answer)
                continue
            _print_answer(result)
            if not interactive or result.get("stop_reason") != "final":
                exit_code = 0 if result.get("stop_reason") == "final" else 2
                break
            pending = _follow_up_input(graph, config, once=False)
        if args.trace and result:
            _write_trace(args.trace, thread_id, result)
            print(f"\ntrace written to {args.trace}")
        return exit_code
    finally:
        flush()
        client.close()
        connection.close()


def _opening(client: McpToolClient, *, message: str | None, incident_id: str) -> str:
    if message is None:
        fetched = client.call_tool("get_incident", {"incident_id": incident_id})
        if fetched.get("status") != 200:
            raise SystemExit(fetched.get("error") or f"Could not read {incident_id}")
        message = incident_user_message(fetched["result"])
    meta_path = Path("data/meta.json")
    now = "2026-09-30T10:20:00Z"
    if meta_path.is_file():
        now = json.loads(meta_path.read_text(encoding="utf-8")).get("simulated_now", now)
    return opening_message(message, now=now, incident_id=incident_id)


def _follow_up_input(graph: Any, config: dict[str, Any], *, once: bool) -> dict[str, Any] | None:
    if once or not sys.stdin.isatty():
        return None
    try:
        follow_up = input("\nYou (empty line to exit): ").strip()
    except EOFError:
        return None
    if not follow_up:
        return None
    messages = list(graph.get_state(config).values.get("messages") or [])
    messages.append(user_text(follow_up))
    return {
        "messages": messages,
        "stop_reason": None,
        "final_response": None,
        "pending_tool_call": None,
        "pending_tool_calls": [],
        "approval_required": False,
        "approval_status": None,
    }


def _read_decision() -> str | None:
    while True:
        try:
            answer = input("Approve? [approve/reject]: ").strip().lower()
        except EOFError:
            return None
        if answer in {"approve", "approved", "yes", "y", "reject", "rejected", "no", "n"}:
            return "approve" if answer in {"approve", "approved", "yes", "y"} else "reject"
        print("Type approve or reject.")


def _format_approval(payload: dict[str, Any]) -> str:
    lines = ["", "=====================================", " HUMAN APPROVAL REQUIRED", "=====================================", ""]
    if payload.get("incident_id"):
        lines.append(f"Incident:\n{payload['incident_id']}\n")
    for action in payload.get("actions") or []:
        lines.append(f"Action:\n{action.get('name')}\n")
        lines.append("Arguments:\n" + json.dumps(action.get("arguments") or {}, indent=2) + "\n")
        if action.get("reason"):
            lines.append(f"Reason:\n{action['reason']}\n")
    lines.append("The action has not been executed.")
    return "\n".join(lines)


def print_step(step: ToolStep, *, verbose: bool) -> None:
    print(f"  -> {step.name} {json.dumps(step.arguments, default=str)}")
    if verbose:
        print(json.dumps(step.response, indent=2, default=str))
    else:
        print(f"     {_summarize(step)}")


def _summarize(step: ToolStep) -> str:
    response = step.response
    if response.get("status") != 200:
        return f"status {response.get('status')}: {response.get('error')}"
    result = response.get("result")
    if not isinstance(result, dict):
        return "status 200"
    if "logs" in result:
        return f"status 200: {result['returned']} log lines, {result['total_matches']} matches"
    if "metrics" in result:
        summary = result.get("summary", {})
        error = (summary.get("error_rate") or {}).get("latest")
        extra = f", latest error rate {error}" if error is not None else ""
        return f"status 200: {summary.get('points')} metric points{extra}"
    if "health_improved" in result:
        return f"status 200: {result.get('result')} health_improved={result['health_improved']}"
    if "healthy_replicas" in result and "status" in result:
        return f"status 200: {result['service']} {result['status']} ({result['healthy_replicas']}/{result['replicas']} replicas, error rate {result.get('error_rate')})"
    if "title" in result and "incident_id" in result:
        return f"status 200: {result['incident_id']} {result['status']} — {result['title']}"
    return "status 200"


def _print_answer(result: dict[str, Any]) -> None:
    print()
    if result.get("stop_reason") == "final":
        print(result.get("final_response") or "(model returned no text)")
    elif result.get("stop_reason") == "max_steps":
        print(f"Stopped after {result.get('step_count')} model turns. Raise --max-steps to continue.")
    else:
        print(f"Stopped: {result.get('stop_reason')}")


def _write_trace(path: Path, thread_id: str, result: dict[str, Any]) -> None:
    payload = {
        "thread_id": thread_id,
        "stop_reason": result.get("stop_reason"),
        "incident_id": result.get("incident_id"),
        "service": result.get("service"),
        "diagnosis": result.get("diagnosis"),
        "remediation": result.get("remediation"),
        "summary": result.get("summary"),
        "step_count": result.get("step_count"),
        "final_response": result.get("final_response"),
        "audit": result.get("audit"),
        "rejected_calls": result.get("rejected_calls"),
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
