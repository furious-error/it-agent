"""Manual Gemini tool loop.

The model only proposes calls. This module executes them with ``ToolKit.execute``
and sends the structured result back, appending the model's original content so
thought signatures stay intact. Automatic function calling is disabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from google import genai
from google.genai import types

from mcp_server.tools import ToolKit, tool_risk

from .config import Settings
from .prompts import SYSTEM_PROMPT
from .schemas import gemini_tool, gemini_tool_from_discovered
from .telemetry import PROPOSE_NEXT_ACTION, generation_io, observation


class Reasoner(Protocol):
    def generate(self, contents: list[Any]) -> Any:
        """One model turn over the conversation so far."""


@dataclass
class ToolStep:
    name: str
    arguments: dict[str, Any]
    response: dict[str, Any]
    risk: str

    @property
    def succeeded(self) -> bool:
        return self.response.get("status") == 200


@dataclass
class TurnResult:
    final_response: str | None
    steps: list[ToolStep]
    stop_reason: str
    model_turns: int


@dataclass
class IncidentAgent:
    """Multi-turn investigator. ``send`` runs until the model stops calling tools."""

    reasoner: Reasoner
    toolkit: ToolKit
    max_steps: int = 12
    on_step: Callable[[ToolStep], None] | None = None
    contents: list[Any] = field(default_factory=list)
    _catalog: dict[str, dict[str, Any]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._catalog = {tool["name"]: tool for tool in self.toolkit.catalog()}

    def send(self, user_message: str) -> TurnResult:
        self.contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
        steps: list[ToolStep] = []
        for turn in range(1, self.max_steps + 1):
            response = self.reasoner.generate(list(self.contents))
            content = _model_content(response)
            if content is None:
                return TurnResult(None, steps, "no_candidate", turn)
            # The model's content is appended unchanged so thought signatures survive.
            self.contents.append(content)
            calls = _function_calls(content)
            if not calls:
                return TurnResult(_visible_text(content) or None, steps, "final", turn)
            response_parts = [self._execute_call(call, steps) for call in calls]
            self.contents.append(types.Content(role="user", parts=response_parts))
        return TurnResult(None, steps, "max_steps", self.max_steps)

    def _execute_call(self, call: Any, steps: list[ToolStep]) -> types.Part:
        arguments = _coerce_arguments(call.name, _as_dict(getattr(call, "args", None)), self._catalog.get(call.name))
        response = self.toolkit.execute(call.name, arguments)
        step = ToolStep(name=call.name, arguments=arguments, response=response, risk=tool_risk(call.name))
        steps.append(step)
        if self.on_step:
            self.on_step(step)
        payload: dict[str, Any] = {"name": call.name, "response": response}
        call_id = getattr(call, "id", None)
        if call_id:
            payload["id"] = call_id
        return types.Part(function_response=types.FunctionResponse(**payload))


class GeminiReasoner:
    """SDK adapter. Tools are schemas, and automatic execution is off."""

    def __init__(self, client: genai.Client, settings: Settings, toolkit: ToolKit | None = None) -> None:
        self.client = client
        self.model = settings.model
        static_tools = [gemini_tool(toolkit.catalog())] if toolkit is not None else []
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=static_tools,
            temperature=settings.temperature,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(thinking_level=settings.thinking_level(), include_thoughts=False),
        )

    def generate(self, contents: list[Any], tools: list[dict[str, Any]] | None = None) -> Any:
        config = self.config
        if tools is not None:
            # Discovered schemas replace the static catalog. Execution stays manual.
            config = self.config.model_copy(update={"tools": [gemini_tool_from_discovered(tools)]})
        with observation(as_type="generation", name=PROPOSE_NEXT_ACTION, model=self.model) as gen:
            response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
            prompt, output, usage = generation_io(contents, response)
            gen.update(
                input=prompt,
                output=output,
                usage_details=usage,
                model=self.model,
                metadata={"tools": [_tool_brief(tool) for tool in tools or []]},
            )
            return response


def _tool_brief(tool: Any) -> dict[str, str]:
    if not isinstance(tool, dict):
        return {"name": "", "description": ""}
    return {"name": str(tool.get("name") or ""), "description": str(tool.get("description") or "")}


def _model_content(response: Any) -> Any | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    return getattr(candidates[0], "content", None)


def _function_calls(content: Any) -> list[Any]:
    calls = []
    for part in getattr(content, "parts", None) or []:
        call = getattr(part, "function_call", None)
        if call is not None and getattr(call, "name", None):
            calls.append(call)
    return calls


def _visible_text(content: Any) -> str:
    chunks = []
    for part in getattr(content, "parts", None) or []:
        if getattr(part, "thought", False):
            continue
        text = getattr(part, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()


def _as_dict(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    return {}


def _coerce_arguments(name: str, arguments: dict[str, Any], spec: dict[str, Any] | None) -> dict[str, Any]:
    """Gemini sometimes emits whole numbers as floats; integer parameters are coerced."""
    if spec is None:
        return arguments
    types_by_name = {param["name"]: param["type"] for param in spec["parameters"]}
    coerced = {}
    for key, value in arguments.items():
        annotation = types_by_name.get(key, "")
        if annotation.split("|")[0].strip() == "int" and isinstance(value, float) and value.is_integer():
            value = int(value)
        coerced[key] = value
    return coerced
