"""Conversation history tool registered on the shared MCP server instance."""

from __future__ import annotations

import json
import logging

from src.tools import mcp

logger = logging.getLogger("tools.memory.conversation")


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
        conversation_id = await store.get_or_create_conversation(user_id, platform, agent_id="jarvis")
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
        logger.error("get_recent_conversation error for user_id=%s: %s", user_id, exc)
        return f"Error retrieving conversation: {exc}"
    finally:
        await store.close()
