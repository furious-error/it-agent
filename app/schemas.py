"""Turn the tool catalog into Gemini function declarations.

Declarations describe the tools. They are not Python callables, so the SDK has
nothing it can execute on its own; ``app.agent`` runs each call through ``ToolKit.execute``.
"""

from __future__ import annotations

from typing import Any

from google.genai import types

from mcp_server.tools import INCIDENT_STATUSES, LOG_LEVELS, SEVERITIES

PARAM_HELP = {
    "incident_id": "Incident id, for example INC-001.",
    "service": "Service name, for example orders-api.",
    "deployment_id": "Deployment id, for example deploy-8472.",
    "start_time": "Inclusive range start, ISO 8601 UTC, for example 2026-09-30T10:00:00Z.",
    "end_time": "Inclusive range end, ISO 8601 UTC. Defaults to the simulated current time.",
    "since": "Only deployments at or after this ISO 8601 UTC timestamp.",
    "keyword": "Case-insensitive substring matched against the log message.",
    "level": "Log level filter.",
    "limit": "Maximum number of records to return.",
    "status": "Status filter.",
    "database": "Database name, for example orders-db. Omit to list every database.",
    "title": "Short incident title.",
    "severity": "Incident severity.",
    "description": "What is going wrong.",
    "note": "Investigation note to append to the incident.",
    "author": "Note author. Defaults to the agent.",
    "root_cause": "Root cause statement to record on the incident.",
    "resolution": "How the incident was resolved.",
    "reason": "Why this action is being taken.",
    "replicas": "Desired replica count, an integer from 1 to 20.",
    "feature_flag": "Name of the feature flag to turn off.",
}

ENUMS = {
    ("search_logs", "level"): sorted(LOG_LEVELS),
    ("list_incidents", "status"): sorted(INCIDENT_STATUSES),
    ("update_incident", "status"): sorted(INCIDENT_STATUSES),
    ("get_alerts", "status"): ["firing", "resolved"],
    ("create_incident", "severity"): sorted(SEVERITIES),
    ("update_incident", "severity"): sorted(SEVERITIES),
}

_JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}


def _json_type(annotation: str) -> str:
    base = annotation.split("|")[0].strip()
    return _JSON_TYPES.get(base, "string")


def discovery_records(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool records in the shape ``list_tools`` returns: name, description, input_schema."""
    records = []
    for tool in catalog:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in tool["parameters"]:
            schema: dict[str, Any] = {
                "type": _json_type(param["type"]),
                "description": PARAM_HELP.get(param["name"], param["name"]),
            }
            enum = ENUMS.get((tool["name"], param["name"]))
            if enum:
                schema["enum"] = enum
            properties[param["name"]] = schema
            if param["required"]:
                required.append(param["name"])
        parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        records.append(
            {
                "name": tool["name"],
                "description": f"[{tool['risk']}] {tool['description']}",
                "input_schema": parameters,
            }
        )
    return records


def function_declarations(catalog: list[dict[str, Any]]) -> list[types.FunctionDeclaration]:
    return [_declaration(record) for record in discovery_records(catalog)]


def declarations_from_discovered(tools: list[dict[str, Any]]) -> list[types.FunctionDeclaration]:
    """Build Gemini declarations from tools the process discovered (MCP ``list_tools``)."""
    return [_declaration(tool) for tool in tools]


def gemini_tool(catalog: list[dict[str, Any]]) -> types.Tool:
    return types.Tool(function_declarations=function_declarations(catalog))


def gemini_tool_from_discovered(tools: list[dict[str, Any]]) -> types.Tool:
    return types.Tool(function_declarations=declarations_from_discovered(tools))


def _declaration(tool: dict[str, Any]) -> types.FunctionDeclaration:
    schema = tool.get("input_schema") or {"type": "object", "properties": {}}
    return types.FunctionDeclaration(
        name=tool["name"],
        description=tool.get("description") or tool["name"],
        parameters_json_schema=schema,
    )
