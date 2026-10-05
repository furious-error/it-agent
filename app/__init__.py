"""Incident-response agent.

The supported path is the LangGraph workflow in ``app.graph``: Gemini proposes
tool calls, an MCP server executes them, hooks gate the dangerous ones, and a
SQLite checkpoint keeps the run resumable.
"""
