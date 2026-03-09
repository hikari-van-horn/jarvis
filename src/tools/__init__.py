"""Jarvis tools package.

This module owns the single shared FastMCP instance (``mcp``) and its
configuration. Tool modules import ``mcp`` from here to register their
``@mcp.tool()`` decorated functions.

Tool registration is triggered by ``src.tools.server``, which explicitly
imports each tool module in the correct order before starting the server.
"""

from os import getenv

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------
MCP_HOST = getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(getenv("MCP_PORT", 8001))

mcp = FastMCP(
    "jarvis-tools",
    instructions=(
        "Tools available to the Jarvis AI assistant. "
        "Use these to look up real-time information or query long-term memory."
    ),
    host=MCP_HOST,
    port=MCP_PORT,
    log_level="WARNING",
)
