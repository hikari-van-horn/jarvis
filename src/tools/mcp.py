"""
Local MCP server for Jarvis.

Exposes tools that the agent can invoke during conversations.
Runs via FastMCP with SSE transport on localhost:8001.

Tools exposed
-------------
- get_current_datetime   : current date/time in any IANA timezone
- get_user_memory        : fetch long-term user persona from SurrealDB
- get_recent_conversation: fetch recent conversation history from SurrealDB

Usage (standalone)::

    python -m src.tools.mcp

Or started programmatically by main.py via a background thread.
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("tools.mcp")

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------
MCP_HOST =  os.getenv('MCP_HOST', "127.0.0.1")
MCP_PORT = int(os.getenv('MCP_PORT', 8001))

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

# ---------------------------------------------------------------------------
# Time & date tools
# ---------------------------------------------------------------------------


@mcp.tool()
def get_current_datetime(timezone: str = "UTC") -> str:
    """Get the current date and time in the specified IANA timezone.

    Args:
        timezone: IANA timezone name, e.g. 'America/New_York', 'Asia/Tokyo', 'UTC'.
    """
    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except ZoneInfoNotFoundError:
        logger.warning("Unknown timezone '%s', falling back to UTC", timezone)
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception as exc:
        logger.error("get_current_datetime error: %s", exc)
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_user_memory(user_id: str) -> str:
    """Retrieve the long-term memory / persona profile for a specific user.

    Args:
        user_id: The unique identifier for the user (e.g. Discord user ID).
    """
    from src.agent.memory.store import MemoryStore

    store = MemoryStore()
    try:
        await store.connect()
        memory = await store.get_user_memory(user_id)
        if memory:
            return json.dumps(memory, ensure_ascii=False, indent=2)
        return f"No memory found for user '{user_id}'."
    except Exception as exc:
        logger.error("get_user_memory error for user_id=%s: %s", user_id, exc)
        return f"Error retrieving memory: {exc}"
    finally:
        await store.close()


@mcp.tool()
async def get_recent_conversation(
    user_id: str,
    platform: str = "discord",
    limit: int = 10,
) -> str:
    """Retrieve recent conversation history for a user.

    Args:
        user_id:  The unique identifier for the user.
        platform: The platform the conversation is on (discord, telegram, …).
        limit:    Maximum number of recent messages to return.
    """
    from langchain_core.messages import HumanMessage
    from src.agent.memory.conversation_store import ConversationStore

    store = ConversationStore()
    try:
        await store.connect()
        conversation_id = await store.get_or_create_conversation(
            user_id, platform, agent_id="jarvis"
        )
        messages = await store.load_as_langchain_messages(conversation_id, limit=limit)
        history = [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in messages
        ]
        return json.dumps(history, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error(
            "get_recent_conversation error for user_id=%s: %s", user_id, exc
        )
        return f"Error retrieving conversation: {exc}"
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Entry point for standalone execution
# ---------------------------------------------------------------------------


def run(host: str = MCP_HOST, port: int = MCP_PORT) -> None:
    """Start the MCP server with SSE transport (blocking)."""
    logger.info("Starting MCP server on %s:%d (SSE transport)", host, port)
    # Reconfigure host/port if caller overrides the defaults.
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport="sse")


if __name__ == "__main__":
    import sys
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Ensure src/ is importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    run()
