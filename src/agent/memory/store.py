"""
SurrealDB-backed memory store for user personas.

Tables used
-----------
- ``user_memory``  : stores the UserPersona JSON document, keyed by user_id.
  Record IDs follow the pattern  ``user_memory:u_<user_id>``  to keep them
  SurrealDB-identifier-safe.
"""

import json
import logging
from typing import Any

import jsonpatch
from surrealdb import AsyncSurreal, RecordID

from src.config import SURREAL_URL, SURREAL_USER, SURREAL_PASS, SURREAL_NS, SURREAL_DB

logger = logging.getLogger("agent.memory.store")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record_id(user_id: str) -> RecordID:
    """Return a SurrealDB RecordID for a given user."""
    safe = user_id.replace(":", "_").replace("`", "")
    return RecordID("user_memory", f"u_{safe}")


# ---------------------------------------------------------------------------
# Store class
# ---------------------------------------------------------------------------

class MemoryStore:
    """Async SurrealDB client wrapper for user-persona CRUD operations.

    Because SurrealDB's HTTP connection is effectively stateless (no persistent
    socket), each call re-authenticates via the pre-configured credentials.
    ``connect()`` authenticates once and caches the token; subsequent calls
    reuse the cached connection object.

    Usage::

        store = MemoryStore()
        await store.connect()
        memory = await store.get_user_memory(user_id)
        ...
        await store.close()

    Or as an async context manager::

        async with MemoryStore() as store:
            ...
    """

    def __init__(self) -> None:
        self._db = None
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create the SurrealDB connection and authenticate (idempotent)."""
        if self._connected and self._db is not None:
            return
        self._db = AsyncSurreal(url=SURREAL_URL)
        await self._db.signin({"username": SURREAL_USER, "password": SURREAL_PASS})
        await self._db.use(SURREAL_NS, SURREAL_DB)
        self._connected = True
        logger.info(
            "MemoryStore connected to SurrealDB [ns=%s db=%s]", SURREAL_NS, SURREAL_DB
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

    async def __aenter__(self) -> "MemoryStore":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    async def init_schema(self) -> None:
        """Execute schema.surql to define or update all tables/indexes.

        Safe to call on every startup — all statements use ``IF NOT EXISTS``.
        """
        import os
        schema_path = os.path.join(os.path.dirname(__file__), "schema.surql")
        try:
            with open(schema_path, "r", encoding="utf-8") as fh:
                ddl = fh.read()
            for stmt in ddl.split(";"):
                stmt = stmt.strip()
                if stmt:
                    await self.db.query(stmt)
            logger.info("MemoryStore: schema applied from %s", schema_path)
        except Exception as exc:
            logger.error("MemoryStore.init_schema failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Internal convenience
    # ------------------------------------------------------------------

    @property
    def db(self):
        if self._db is None or not self._connected:
            raise RuntimeError("MemoryStore is not connected. Call connect() first.")
        return self._db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_user_memory(self, user_id: str) -> dict | None:
        """Retrieve the persisted UserPersona for *user_id*, or ``None`` if not found."""
        try:
            result = await self.db.select(_make_record_id(user_id))
            if result:
                if isinstance(result, list):
                    return result[0] if result else None
                return result
        except Exception as exc:
            logger.warning("get_user_memory failed for user_id=%s: %s", user_id, exc)
        return None

    async def upsert_user_memory(self, user_id: str, data: dict) -> dict:
        """Create or fully replace the UserPersona record for *user_id*."""
        try:
            data = dict(data)
            data["user_id"] = user_id
            result = await self.db.upsert(_make_record_id(user_id), data)
            logger.info("upsert_user_memory succeeded for user_id=%s", user_id)
            if isinstance(result, list):
                return result[0] if result else data
            return result if isinstance(result, dict) else data
        except Exception as exc:
            logger.error("upsert_user_memory failed for user_id=%s: %s", user_id, exc)
            raise

    async def apply_patches(self, user_id: str, patches: list[dict]) -> dict:
        """Apply a JSON Patch (RFC 6902) list to the stored UserPersona.

        Returns the updated document.
        """
        if not patches:
            logger.debug(
                "apply_patches: empty patch list for user_id=%s, skipping", user_id
            )
            return await self.get_user_memory(user_id) or {}

        current = await self.get_user_memory(user_id)
        if current is None:
            logger.warning(
                "apply_patches: no existing memory for user_id=%s, starting from empty",
                user_id,
            )
            current = {}

        # Strip the SurrealDB internal 'id' field before patching
        current.pop("id", None)

        try:
            patch = jsonpatch.JsonPatch(patches)
            updated = patch.apply(current)
            logger.info(
                "apply_patches: applied %d operations to user_id=%s",
                len(patches),
                user_id,
            )
        except jsonpatch.JsonPatchException as exc:
            logger.error(
                "apply_patches: patch failed for user_id=%s: %s", user_id, exc
            )
            raise

        return await self.upsert_user_memory(user_id, updated)

    async def ensure_user_memory(self, user_id: str, default: dict) -> dict:
        """Return existing memory or initialise with *default* if absent."""
        existing = await self.get_user_memory(user_id)
        if existing:
            return existing
        logger.info(
            "ensure_user_memory: initialising memory for user_id=%s", user_id
        )
        return await self.upsert_user_memory(user_id, default)
