"""Context compaction. This limits how much transcript is sent to the model.

It is not checkpointing. A checkpoint restores the workflow after a process
restart; compaction only replaces old messages with a summary.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from google.genai import types

from .messages import render_message
from .telemetry import SUMMARIZE_CONTEXT, generation_io, observation

# The plan's rule: once the transcript is longer than 10 messages, summarize the oldest ones.
COMPACTION_THRESHOLD = 10
SUMMARIZE_PREFIX = 8

SUMMARY_PROMPT = """\
You are compacting the transcript of an SRE incident investigation.

Summarize the earlier messages so another agent can continue the investigation.
Keep the incident id, the service, the evidence from logs, metrics, deployments
and health checks, the actions already taken, their results, and any approval
that is still pending.

Use only facts present in the transcript. Be concise.
"""


class Summarizer(Protocol):
    def summarize(self, transcript: str) -> str:
        """Return a plain-text summary of the transcript."""


class GeminiSummarizer:
    """Separate model call with no tools. ``model`` can be a smaller model than the investigator."""

    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def summarize(self, transcript: str) -> str:
        with observation(as_type="generation", name=SUMMARIZE_CONTEXT, model=self.model) as gen:
            response = self.client.models.generate_content(
                model=self.model,
                contents=transcript,
                config=types.GenerateContentConfig(system_instruction=SUMMARY_PROMPT, temperature=0.2),
            )
            text = (getattr(response, "text", None) or "").strip()
            _, _, usage = generation_io([{"role": "user", "parts": [{"text": transcript}]}], response)
            gen.update(input={"transcript": transcript}, output=text, usage_details=usage, model=self.model)
            return text


def compact_messages(
    messages: list[dict[str, Any]],
    summarize: Callable[[str], str],
    *,
    existing_summary: str | None = None,
    threshold: int = COMPACTION_THRESHOLD,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Return ``(summary, messages)``. Below the threshold the transcript is unchanged."""
    if len(messages) <= threshold:
        return existing_summary, messages
    cut = _safe_cut(messages, SUMMARIZE_PREFIX)
    if cut <= 0 or cut >= len(messages):
        return existing_summary, messages
    rendered = "\n\n".join(render_message(message) for message in messages[:cut])
    if existing_summary:
        rendered = f"Previous summary:\n{existing_summary}\n\nNew messages:\n{rendered}"
    summary = summarize(rendered).strip()
    summary_message = {"role": "user", "parts": [{"text": f"Summary of the investigation so far:\n{summary}"}]}
    return summary, [summary_message, *messages[cut:]]


def _safe_cut(messages: list[dict[str, Any]], target: int) -> int:
    """Index where the summarized prefix ends. Do not leave a function response without its call."""
    cut = min(target, len(messages) - 1)
    while cut < len(messages) and _is_function_response(messages[cut]):
        cut += 1
    if cut <= 0 or cut >= len(messages):
        return 0
    return cut


def _is_function_response(message: dict[str, Any]) -> bool:
    return any(isinstance(part, dict) and "function_response" in part for part in message.get("parts") or [])
