"""MCP server for the simulated AcmeCloud infrastructure.

The process is started with ``INCIDENT_DB`` pointing at a SQLite file. Tools are
the functions in ``mcp_server.tools``; this module only exposes them. Stdout is
the JSON-RPC channel, so nothing here prints.
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import Any, get_type_hints

from mcp.server.mcpserver import MCPServer

from .data_store import DataStore
from .tools import ToolKit, tool_risk


def build_server(db_path: str) -> MCPServer:
    toolkit = ToolKit(DataStore(db_path=db_path))
    server = MCPServer(
        "it-operations-server",
        instructions="Simulated AcmeCloud production infrastructure: services, deployments, logs, metrics, databases, and incidents.",
        log_level="ERROR",
    )
    for name, function in toolkit.registry().items():
        server.tool(name=name, description=f"[{tool_risk(name)}] {(function.__doc__ or '').strip()}")(_expose(toolkit, name, function))
    return server


def main() -> None:
    db_path = os.environ.get("INCIDENT_DB")
    if not db_path:
        print("INCIDENT_DB is required", file=sys.stderr)
        raise SystemExit(2)
    build_server(db_path).run(transport="stdio")


def _expose(toolkit: ToolKit, name: str, function: Any) -> Any:
    signature = inspect.signature(function)
    hints = get_type_hints(function)

    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        bound = signature.bind(*args, **kwargs)
        with toolkit._call_lock:
            return toolkit.execute(name, dict(bound.arguments))

    wrapper.__name__ = name
    wrapper.__doc__ = function.__doc__
    wrapper.__signature__ = signature  # type: ignore[attr-defined]
    wrapper.__annotations__ = hints
    return wrapper


if __name__ == "__main__":
    main()
