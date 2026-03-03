"""
SurrealDB-backed short-term memory: conversation sessions and messages.

Tables used (defined in schema.surql)
--------------------------------------
- ``conversation``  : one record per chat session (user × platform).
  A new session starts when the user returns after ``NEW_SESSION_GAP_HOURS``
  of inactivity.

- ``message``       : individual turns inside a conversation, ordered by
  ``created_at``.  Linked to ``conversation`` via a typed record link.

Typical call sequence
---------------------
::

    async with ConversationStore() as store:
        conv_id = await store.get_or_create_conversation(user_id, "discord")
        await store.append_message(conv_id, "user", user_text)
        history = await store.load_as_langchain_messages(conv_id, limit=20)
        # … run LLM …
        await store.append_message(conv_id, "assistant", reply_text)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from surrealdb import AsyncSurreal, RecordID

from src.config import SURREAL_URL, SURREAL_USER, SURREAL_PASS, SURREAL_NS, SURREAL_DB

logger = logging.getLogger("agent.memory.conversation")

# How many hours of silence before a new session is started.
NEW_SESSION_GAP_HOURS: int = int(os.getenv("JARVIS_SESSION_GAP_HOURS", "6"))

# Maximum byte size of the conversation history injected into the LLM context.
# Messages are trimmed FIFO (oldest dropped first) until the total fits.
CONTEXT_SIZE_LIMIT_BYTES: int = int(os.getenv("JARVIS_CONTEXT_SIZE_BYTES", str(4 * 1024)))

# How many messages to load by default when building LLM context.
DEFAULT_HISTORY_LIMIT: int = 50

# Default agent identifier used when none is supplied.
DEFAULT_AGENT_ID: str = "jarvis"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _role_to_langchain(role: str, content: str) -> BaseMessage:
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    return SystemMessage(content=content)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ConversationStore:
    """Async SurrealDB client for short-term (conversation) memory.

    Mirrors the connection-management pattern of ``MemoryStore`` so both stores
    can be used side-by-side with the same lifecycle conventions.
    """

    def __init__(self) -> None:
        self._db: AsyncSurreal | None = None
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open SurrealDB connection and authenticate (idempotent)."""
        if self._connected and self._db is not None:
            return
        self._db = AsyncSurreal(url=SURREAL_URL)
        await self._db.signin({"username": SURREAL_USER, "password": SURREAL_PASS})
        await self._db.use(SURREAL_NS, SURREAL_DB)
        self._connected = True
        logger.info(
            "ConversationStore connected to SurrealDB [ns=%s db=%s]",
            SURREAL_NS,
            SURREAL_DB,
        )

    async def close(self) -> None:
        try:
            if self._db and hasattr(self._db, "close"):
                await self._db.close()
        except Exception:
            pass
        finally:
            self._db = None
            self._connected = False

    async def __aenter__(self) -> "ConversationStore":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    @property
    def db(self) -> AsyncSurreal:
        if self._db is None or not self._connected:
            raise RuntimeError(
                "ConversationStore is not connected. Call connect() first."
            )
        return self._db

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    async def init_schema(self) -> None:
        """Execute schema.surql to define or update tables/indexes.

        Safe to call on every startup — all statements use ``IF NOT EXISTS``.
        """
        schema_path = os.path.join(os.path.dirname(__file__), "schema.surql")
        try:
            with open(schema_path, "r", encoding="utf-8") as fh:
                ddl = fh.read()
            # Split on statement boundaries and execute each non-empty statement.
            for stmt in ddl.split(";"):
                stmt = stmt.strip()
                if stmt:
                    await self.db.query(stmt)
            logger.info("ConversationStore: schema applied from %s", schema_path)
        except Exception as exc:
            logger.error("ConversationStore.init_schema failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Conversation session management
    # ------------------------------------------------------------------

    async def get_or_create_conversation(
        self,
        user_id: str,
        platform: str,
        agent_id: str = 'jarvis',
        metadata: dict | None = None,
        gap_hours: int = NEW_SESSION_GAP_HOURS,
    ) -> RecordID:
        """Return the current active conversation ID for *(user_id, platform)*.

        If the most-recent session is older than *gap_hours* (or no session
        exists yet), a new conversation record is created.

        Parameters
        ----------
        user_id:
            Platform-agnostic user identifier.
        platform:
            Source platform string, e.g. ``"discord"`` or ``"telegram"``.
        metadata:
            Optional free-form dict stored on the conversation record
            (e.g. ``{"channel_id": "...", "guild_id": "..."}``.
        gap_hours:
            Inactivity threshold in hours before starting a fresh session.

        Returns
        -------
        RecordID
            The SurrealDB record ID of the active conversation.
        """
        # Find the most recently active conversation for this user + platform.
        rows = await self.db.query(
            """
            SELECT id, last_active_at
            FROM conversation
            WHERE user_id = $user_id AND platform = $platform AND agent_id = $agent_id
            ORDER BY last_active_at DESC
            LIMIT 1
            """,
            {"user_id": user_id, "platform": platform, "agent_id": agent_id},
        )

        recent = _first_row(rows)
        if recent:
            last_active = _parse_dt(recent.get("last_active_at"))
            age_hours = (
                (datetime.now(tz=timezone.utc) - last_active).total_seconds() / 3600
                if last_active
                else gap_hours + 1  # treat unparseable as expired
            )
            if age_hours < gap_hours:
                logger.debug(
                    "get_or_create_conversation: reusing session %s (age=%.1fh)",
                    recent["id"],
                    age_hours,
                )
                return recent["id"]

        # Create a new conversation session.
        # now_iso = datetime.now(tz=timezone.utc).isoformat()
        new_rows = await self.db.query(
            """
            CREATE conversation CONTENT {
                user_id:        $user_id,
                agent_id:       $agent_id,
                platform:       $platform,
                started_at:     time::now(),
                last_active_at: time::now(),
                metadata:       $metadata
            } RETURN id
            """,
            {
                "user_id": user_id,
                "platform": platform,
                "agent_id": agent_id,
                "metadata": metadata,
            },
        )
        new_row = _first_row(new_rows)
        if not new_row:
            raise RuntimeError(
                f"ConversationStore: failed to create conversation for user_id={user_id}"
            )
        conv_id = new_row["id"]
        logger.info(
            "get_or_create_conversation: created new session %s for user_id=%s platform=%s",
            conv_id,
            user_id,
            platform,
        )
        return conv_id

    async def touch_conversation(self, conversation_id: RecordID) -> None:
        """Update ``last_active_at`` on the conversation to now."""
        await self.db.query(
            "UPDATE $id SET last_active_at = time::now()",
            {"id": conversation_id},
        )

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------

    async def append_message(
        self,
        conversation_id: RecordID,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> RecordID:
        """Append one message to *conversation_id* and touch the session timestamp.

        Parameters
        ----------
        role:
            One of ``"user"``, ``"assistant"``, or ``"system"``.
        content:
            Raw message text.
        metadata:
            Optional dict (e.g. ``{"model": "deepseek-chat", "tokens": 42}``).
        """
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"append_message: invalid role {role!r}")

        # Extract user_id from conversation record to denormalise.
        conv_rows = await self.db.query(
            "SELECT user_id FROM $id",
            {"id": conversation_id},
        )
        conv = _first_row(conv_rows)
        user_id = conv.get("user_id", "") if conv else ""

        rows = await self.db.query(
            """
            CREATE message CONTENT {
                conversation: $conv_id,
                user_id:      $user_id,
                role:         $role,
                content:      $content,
                created_at:   time::now(),
                metadata:     $metadata
            } RETURN id
            """,
            {
                "conv_id": conversation_id,
                "user_id": user_id,
                "role": role,
                "content": content,
                "metadata": metadata,
            },
        )
        row = _first_row(rows)
        if not row:
            raise RuntimeError("append_message: SurrealDB returned no record")

        # Keep conversation.last_active_at fresh.
        await self.touch_conversation(conversation_id)

        logger.debug(
            "append_message: [%s] %s -> %s (conv=%s)",
            row["id"],
            role,
            content[:60],
            conversation_id,
        )
        return row["id"]

    async def get_recent_messages(
        self,
        conversation_id: RecordID,
        limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> list[dict]:
        """Return the *limit* most-recent messages as raw dicts, oldest first.

        Each dict contains at least: ``id``, ``role``, ``content``, ``created_at``.
        """
        rows = await self.db.query(
            """
            SELECT id, role, content, created_at, metadata
            FROM message
            WHERE conversation = $conv_id
            ORDER BY created_at DESC
            LIMIT $limit
            """,
            {"conv_id": conversation_id, "limit": limit},
        )
        messages = _all_rows(rows)
        # Reverse so the result is chronological (oldest → newest).
        messages.reverse()
        return messages

    async def load_as_langchain_messages(
        self,
        conversation_id: RecordID,
        limit: int = DEFAULT_HISTORY_LIMIT,
        context_size_bytes: int = CONTEXT_SIZE_LIMIT_BYTES,
    ) -> list[BaseMessage]:
        """Load recent messages, apply 4 KB FIFO context window, and return LangChain objects.

        All messages are permanently stored in SurrealDB.  This method only
        controls what is injected into the LLM context window.

        Strategy
        --------
        1. Fetch up to *limit* most-recent messages from DB (already chronological).
        2. Walk the list from **newest to oldest**, accumulating ``role: content``
           byte sizes until the total would exceed *context_size_bytes*.
        3. Keep only the suffix that fits (FIFO — oldest entries are evicted first).
        """
        raw = await self.get_recent_messages(conversation_id, limit=limit)
        if not raw:
            return []

        # Walk newest → oldest, greedily accumulate messages that fit in the budget.
        kept: list[dict] = []
        total_bytes = 0
        for msg in reversed(raw):          # newest first
            # Estimate byte size: role + ": " + content, UTF-8 encoded.
            chunk = f"{msg['role']}: {msg['content']}"
            size = len(chunk.encode("utf-8"))
            if total_bytes + size > context_size_bytes:
                break                      # FIFO: stop; everything older is evicted
            kept.append(msg)
            total_bytes += size

        kept.reverse()                     # restore chronological order (oldest → newest)

        evicted = len(raw) - len(kept)
        if evicted:
            logger.debug(
                "load_as_langchain_messages: evicted %d old message(s) "
                "(budget=%d bytes, used=%d bytes, conv=%s)",
                evicted, context_size_bytes, total_bytes, conversation_id,
            )

        return [_role_to_langchain(m["role"], m["content"]) for m in kept]

    async def get_conversation_stats(self, conversation_id: RecordID) -> dict:
        """Return basic statistics for a conversation (message count, first/last ts)."""
        rows = await self.db.query(
            """
            SELECT
                count()         AS total_messages,
                math::min(created_at) AS first_message_at,
                math::max(created_at) AS last_message_at
            FROM message
            WHERE conversation = $conv_id
            GROUP ALL
            """,
            {"conv_id": conversation_id},
        )
        return _first_row(rows) or {}

    async def list_conversations(
        self, user_id: str, platform: str | None = None, agent_id: str | None = None, limit: int = 20
    ) -> list[dict]:
        """List recent conversations for a user, newest first."""
        filters = ["user_id = $user_id"]
        params: dict = {"user_id": user_id, "limit": limit}
        if platform:
            filters.append("platform = $platform")
            params["platform"] = platform
        if agent_id:
            filters.append("agent_id = $agent_id")
            params["agent_id"] = agent_id
        where = " AND ".join(filters)
        rows = await self.db.query(
            f"""
            SELECT id, agent_id, platform, started_at, last_active_at, metadata
            FROM conversation
            WHERE {where}
            ORDER BY last_active_at DESC
            LIMIT $limit
            """,
            params,
        )
        return _all_rows(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _first_row(result: Any) -> dict | None:
    """Extract the first row from a SurrealDB query result."""
    if result is None:
        return None
    # AsyncSurreal.query returns list[list[dict]] or list[dict]
    if isinstance(result, list):
        if not result:
            return None
        first = result[0]
        if isinstance(first, list):
            return first[0] if first else None
        if isinstance(first, dict):
            return first
    if isinstance(result, dict):
        return result
    return None


def _all_rows(result: Any) -> list[dict]:
    """Flatten a SurrealDB query result into a list of dicts."""
    if result is None:
        return []
    if isinstance(result, list):
        if not result:
            return []
        first = result[0]
        if isinstance(first, list):
            return first
        if isinstance(first, dict):
            return result
    return []


def _parse_dt(value: Any) -> datetime | None:
    """Best-effort parse of a SurrealDB datetime value."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None
