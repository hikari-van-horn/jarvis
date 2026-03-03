"""User knowledge / persona memory tool registered on the shared MCP server instance."""

from __future__ import annotations

import json
import logging

from src.tools import mcp

logger = logging.getLogger("tools.memory.knowledge")


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
async def update_user_memory(user_id: str, patches: str) -> str:
    """Apply RFC 6902 JSON Patch operations to update a user's long-term memory profile.

    Use this tool when the user shares memorable facts such as their job, location,
    preferences, goals, or any other stable personal information.

    The UserPersona document has these top-level keys:
      - demographics  : preferred_name, home_location {city, country_code, timezone},
                        education_history (array), meta
      - preferences   : languages (array), coding_preferences (map),
                        dietary_restrictions (array), daily_life {hobbies (array)}, meta
      - work_context  : roles (array of {title, organization, is_current, description}),
                        expertise_tags (array), meta
      - financial_profile : net_worth_bracket, investment_interests (array), meta

    Args:
        user_id: The unique identifier for the user.
        patches: A JSON string containing an RFC 6902 patch array, e.g.:
                 '[{"op":"replace","path":"/work_context/expertise_tags/0","value":"AI"},
                   {"op":"add","path":"/preferences/languages/-","value":"Python"}]'
                 Use "add" to append to arrays (path ending in "/-"), "replace" to
                 overwrite existing values, and "remove" to delete a field.
    """
    from src.agent.memory.store import MemoryStore

    store = MemoryStore()
    try:
        patch_list = json.loads(patches)
        if not isinstance(patch_list, list):
            return "Error: patches must be a JSON array of RFC 6902 operations."
        await store.connect()
        updated = await store.apply_patches(user_id, patch_list)
        logger.info(
            "update_user_memory: applied %d patches for user_id=%s", len(patch_list), user_id
        )
        return json.dumps(updated, ensure_ascii=False, indent=2)
    except json.JSONDecodeError as exc:
        return f"Error: invalid JSON in patches — {exc}"
    except Exception as exc:
        logger.error("update_user_memory error for user_id=%s: %s", user_id, exc)
        return f"Error updating memory: {exc}"
    finally:
        await store.close()
