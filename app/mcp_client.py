"""Synchronous MCP client over stdio.

LangGraph nodes are synchronous. The client owns one background event loop and
one server process for the life of the investigation. A new process with the
same ``INCIDENT_DB`` file sees the same simulated infrastructure.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, TextIO

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .schemas import discovery_records


class InProcessToolClient:
    """Tool client backed by a local ``ToolKit``. Graph tests use this instead of a subprocess."""

    def __init__(self, toolkit: Any) -> None:
        self.toolkit = toolkit
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_tools(self) -> list[dict[str, Any]]:
        return discovery_records(self.toolkit.catalog())

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return self.toolkit.execute(name, arguments)

    def close(self) -> None:
        return None


class McpToolClient:
    """Spawn ``python -m mcp_server.server`` and talk to it over stdio."""

    def __init__(self, db_path: str | Path, *, cwd: str | Path | None = None, errlog: TextIO | None = None) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._cwd = str(cwd or Path(__file__).resolve().parent.parent)
        self._errlog = errlog
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-client", daemon=True)
        self._thread.start()
        self._stdio = None
        self._session_cm = None
        self._session: ClientSession | None = None
        try:
            self._run(self._connect())
        except Exception:
            self.close()
            raise

    def list_tools(self) -> list[dict[str, Any]]:
        listed = self._run(self._list_tools())
        return [_normalize(tool) for tool in listed]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._run(self._session.call_tool(name, arguments))  # type: ignore[union-attr]
        return _unwrap(result)

    def close(self) -> None:
        if self._session is not None:
            try:
                self._run(self._shutdown())
            except Exception:
                pass
            self._session = None
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    def _run(self, coro: Any, timeout: float = 60) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)

    async def _connect(self) -> None:
        env = {key: value for key, value in os.environ.items() if isinstance(value, str)}
        env["INCIDENT_DB"] = self.db_path
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            env=env,
            cwd=self._cwd,
            encoding="utf-8",
        )
        self._stdio = stdio_client(params, errlog=self._errlog) if self._errlog is not None else stdio_client(params)
        read, write = await self._stdio.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()

    async def _list_tools(self) -> list[Any]:
        assert self._session is not None
        collected = []
        cursor = None
        while True:
            kwargs = {}
            if cursor:
                from mcp import types

                kwargs["params"] = types.PaginatedRequestParams(cursor=cursor)
            result = await self._session.list_tools(**kwargs)
            collected.extend(result.tools)
            cursor = getattr(result, "next_cursor", None) or getattr(result, "nextCursor", None)
            if not cursor:
                return collected

    async def _shutdown(self) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
            self._session_cm = None
        if self._stdio is not None:
            await self._stdio.__aexit__(None, None, None)
            self._stdio = None


def _normalize(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
    if hasattr(schema, "model_dump"):
        schema = schema.model_dump(by_alias=True, exclude_none=True)
    return {
        "name": tool.name,
        "description": (getattr(tool, "description", None) or "").strip(),
        "input_schema": _clean_schema(schema if isinstance(schema, dict) else {}),
    }


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    properties = {}
    for name, prop in (schema.get("properties") or {}).items():
        if not isinstance(prop, dict):
            properties[name] = {"type": "string"}
            continue
        properties[name] = {"type": _property_type(prop), "description": prop.get("description") or name}
        if prop.get("enum"):
            properties[name]["enum"] = list(prop["enum"])
    cleaned: dict[str, Any] = {"type": "object", "properties": properties}
    if schema.get("required"):
        cleaned["required"] = list(schema["required"])
    return cleaned


def _property_type(prop: dict[str, Any]) -> str:
    if isinstance(prop.get("type"), str):
        return prop["type"]
    for option in prop.get("anyOf") or []:
        if isinstance(option, dict) and option.get("type") not in (None, "null"):
            return option["type"]
    return "string"


def _as_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """MCP structured output wraps a dict return value as ``{"result": <value>}``."""
    inner = payload.get("result")
    if "status" not in payload and isinstance(inner, dict) and "status" in inner:
        return inner
    if "status" in payload:
        return payload
    return {"status": 200, "result": payload}


def _unwrap(result: Any) -> dict[str, Any]:
    if getattr(result, "is_error", False):
        return {"status": 500, "error": _text(result) or "tool failed"}
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return _as_envelope(structured)
    text = _text(result)
    if not text:
        return {"status": 500, "error": "empty tool result"}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"status": 500, "error": text}
    if isinstance(parsed, dict) and "status" in parsed:
        return parsed
    return {"status": 200, "result": parsed}


def _text(result: Any) -> str:
    chunks = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()
