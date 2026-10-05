"""Serialize Gemini contents so the checkpoint can store them and the next turn can restore them."""

from __future__ import annotations

import base64
from typing import Any

from google.genai import types


def user_text(text: str) -> dict[str, Any]:
    return {"role": "user", "parts": [{"text": text}]}


def content_to_dict(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if hasattr(content, "model_dump"):
        return content.model_dump(mode="json", exclude_none=True)
    parts = []
    for part in getattr(content, "parts", None) or []:
        dumped: dict[str, Any] = {}
        text = getattr(part, "text", None)
        if text:
            dumped["text"] = text
        if getattr(part, "thought", False):
            dumped["thought"] = True
        call = getattr(part, "function_call", None)
        if call is not None and getattr(call, "name", None):
            dumped["function_call"] = {
                "name": call.name,
                "args": _plain_args(getattr(call, "args", None)),
            }
            call_id = getattr(call, "id", None)
            if call_id:
                dumped["function_call"]["id"] = call_id
        signature = getattr(part, "thought_signature", None)
        if signature:
            dumped["thought_signature"] = base64.b64encode(signature).decode("ascii") if isinstance(signature, bytes) else signature
        if dumped:
            parts.append(dumped)
    return {"role": getattr(content, "role", None) or "model", "parts": parts}


def content_from_dict(data: dict[str, Any]) -> types.Content:
    try:
        return types.Content.model_validate(data)
    except Exception:
        cooked = _decode_signatures(data)
        return types.Content.model_validate(cooked)


def calls_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for part in message.get("parts") or []:
        call = part.get("function_call") or part.get("functionCall")
        if not isinstance(call, dict) or not call.get("name"):
            continue
        calls.append(
            {
                "name": call["name"],
                "arguments": dict(call.get("args") or call.get("arguments") or {}),
                "id": call.get("id"),
            }
        )
    return calls


def function_response_message(responses: list[dict[str, Any]]) -> dict[str, Any]:
    parts = []
    for item in responses:
        body: dict[str, Any] = {"name": item["name"], "response": item["response"]}
        if item.get("id"):
            body["id"] = item["id"]
        parts.append({"function_response": body})
    return {"role": "user", "parts": parts}


def visible_text(message: dict[str, Any]) -> str:
    chunks = []
    for part in message.get("parts") or []:
        if part.get("thought"):
            continue
        if part.get("text"):
            chunks.append(part["text"])
    return "\n".join(chunks).strip()


def coerce_arguments(arguments: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
    """Whole-number floats become integers when the discovered schema says integer."""
    properties = (schema or {}).get("properties") or {}
    coerced = {}
    for key, value in arguments.items():
        expected = (properties.get(key) or {}).get("type")
        if expected == "integer" and isinstance(value, float) and value.is_integer():
            value = int(value)
        coerced[key] = value
    return coerced


def render_message(message: dict[str, Any]) -> str:
    """Plain-text view of one message for the summarizer. Omits thought signatures."""
    lines = [f"[{message.get('role', 'unknown')}]"]
    for part in message.get("parts") or []:
        if part.get("thought"):
            continue
        if part.get("text"):
            lines.append(part["text"])
        call = part.get("function_call")
        if isinstance(call, dict):
            lines.append(f"tool call {call.get('name')} {call.get('args') or {}}")
        response = part.get("function_response")
        if isinstance(response, dict):
            lines.append(f"tool result {response.get('name')}: {_short(response.get('response'))}")
    return "\n".join(lines)


def _plain_args(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return {str(key): item for key, item in value.items()}
    return {}


def _decode_signatures(data: dict[str, Any]) -> dict[str, Any]:
    cooked = dict(data)
    parts = []
    for part in data.get("parts") or []:
        part = dict(part)
        signature = part.get("thought_signature")
        if isinstance(signature, str):
            try:
                part["thought_signature"] = base64.b64decode(signature)
            except Exception:
                part.pop("thought_signature", None)
        parts.append(part)
    cooked["parts"] = parts
    return cooked


def _short(value: Any, limit: int = 500) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."
