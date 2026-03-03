"""
MCP server entry point for Jarvis.

Responsibilities
----------------
1. Import all tool modules so their @mcp.tool() decorators fire and register
   the tools onto the shared FastMCP instance (defined in src.tools.__init__).
2. Expose run() which starts the server (blocking, SSE transport).

Import order is intentional and must stay this way to avoid circular imports:

    src.tools              ← creates the FastMCP `mcp` instance (no child imports)
        ↓
    src.tools.datetime_tools     ← imports `mcp` from src.tools, registers tools
    src.tools.memory.knowledge   ← imports `mcp` from src.tools, registers tools
    src.tools.memory.conversation← imports `mcp` from src.tools, registers tools
        ↓
    server.run()           ← calls mcp.run() with all tools already registered

Usage (standalone)::

    python -m src.tools.server

Or started programmatically by main.py via a background thread.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("tools.server")

# ---------------------------------------------------------------------------
# Tool registration
# Import each tool module so @mcp.tool() decorators execute.
# src.tools.__init__ (which owns `mcp`) is always imported first by Python
# before any of these sub-imports, so there is no circular dependency.
# ---------------------------------------------------------------------------
import src.tools.datetime_tools          # noqa: F401  – registers get_current_datetime
import src.tools.memory.knowledge        # noqa: F401  – registers get_user_memory
import src.tools.memory.conversation     # noqa: F401  – registers get_recent_conversation

# Pull shared instance + config after tool modules are loaded.
from src.tools import mcp, MCP_HOST, MCP_PORT

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(host: str = MCP_HOST, port: int = MCP_PORT) -> None:
    """Start the MCP server with SSE transport (blocking).

    Wraps the FastMCP Starlette app with CORSMiddleware so the MCP inspector
    (and any browser-based client) can connect without CORS errors.
    """
    import anyio
    import uvicorn
    from starlette.middleware.cors import CORSMiddleware

    logger.info("Starting MCP server on %s:%d (SSE transport)", host, port)
    mcp.settings.host = host
    mcp.settings.port = port

    starlette_app = mcp.sse_app()
    app_with_cors = CORSMiddleware(
        starlette_app,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    config = uvicorn.Config(
        app_with_cors,
        host=host,
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    anyio.run(server.serve)


if __name__ == "__main__":
    import sys
    import os
    import logging as _logging

    _logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Ensure project root is on sys.path when run directly.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    run()
