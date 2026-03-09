"""
SurrealDB-backed memory store for user personas.

Tables used
-----------
- ``user_memory``  : stores the UserPersona JSON document, keyed by user_id.
  Record IDs follow the pattern  ``user_memory:u_<user_id>``  to keep them
  SurrealDB-identifier-safe.
"""

import logging
from datetime import date, datetime
from typing import Any

import jsonpatch
from surrealdb import AsyncSurreal, RecordID

from src.config import SURREAL_DB, SURREAL_NS, SURREAL_PASS, SURREAL_URL, SURREAL_USER

logger = logging.getLogger("agent.memory.store")


# ---------------------------------------------------------------------------
# Empty document skeleton — mirrors UserPersona proto structure.
# Used to pre-seed new records so JSON Patch ops on nested paths never
# fail with "member not found".
# ---------------------------------------------------------------------------
_PERSONA_SKELETON: dict = {
    "demographics": {
        "preferred_name": "",
        "home_location": {"city": "", "country_code": "", "timezone": ""},
        "education_history": [],
        "meta": {},
    },
    "preferences": {
        "languages": [],
        "coding_preferences": {},
        "dietary_restrictions": [],
        "daily_life": {"hobbies": []},
        "meta": {},
    },
    "work_context": {
        "roles": [],
        "expertise_tags": [],
        "meta": {},
    },
    "financial_profile": {
        "net_worth_bracket": "",
        "investment_interests": [],
        "meta": {},
    },
}


def _safe_apply_patches(doc: dict, patches: list[dict]) -> dict:
    """Apply RFC 6902 patches one at a time with best-effort fallbacks.

    For each operation:
    - Try applying as-is.
    - If it fails and the op is ``replace``, retry as ``add`` (handles paths
      that don't exist yet despite the skeleton).
    - If still failing, log a warning and skip that single op so the rest
      of the patch set is not blocked.
    """
    import copy

    result = copy.deepcopy(doc)
    for op in patches:
        try:
            result = jsonpatch.JsonPatch([op]).apply(result)
        except jsonpatch.JsonPatchException as exc:
            if op.get("op") == "replace":
                fallback = dict(op, op="add")
                try:
                    result = jsonpatch.JsonPatch([fallback]).apply(result)
                    logger.debug("_safe_apply: converted replace→add for path %s", op.get("path"))
                    continue
                except jsonpatch.JsonPatchException:
                    pass
            logger.warning("_safe_apply: skipping op %s — %s", op, exc)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record_id(user_id: str) -> RecordID:
    """Return a SurrealDB RecordID for a given user."""
    safe = user_id.replace(":", "_").replace("`", "")
    return RecordID("user_memory", f"u_{safe}")


def _strip_record_ids(obj):
    """Recursively sanitise a SurrealDB document for JSON serialisation.

    - Removes the top-level ``id`` key (SurrealDB internal RecordID).
    - Converts any ``RecordID`` values to their string representation.
    - Converts ``datetime`` / ``date`` objects to ISO-8601 strings.
    """
    if isinstance(obj, RecordID):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _strip_record_ids(v) for k, v in obj.items() if k not in ("id", "updated_at")}
    if isinstance(obj, list):
        return [_strip_record_ids(v) for v in obj]
    return obj


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
        logger.info("MemoryStore connected to SurrealDB [ns=%s db=%s]", SURREAL_NS, SURREAL_DB)

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
                raw = result[0] if isinstance(result, list) else result
                if raw:
                    return _strip_record_ids(raw)
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
            raw = result[0] if isinstance(result, list) else result
            return _strip_record_ids(raw) if isinstance(raw, dict) else _strip_record_ids(data)
        except Exception as exc:
            logger.error("upsert_user_memory failed for user_id=%s: %s", user_id, exc)
            raise

    async def apply_patches(self, user_id: str, patches: list[dict]) -> dict:
        """Apply a JSON Patch (RFC 6902) list to the stored UserPersona.

        Returns the updated document.
        """
        if not patches:
            logger.debug("apply_patches: empty patch list for user_id=%s, skipping", user_id)
            return await self.get_user_memory(user_id) or {}

        current = await self.get_user_memory(user_id)
        if current is None:
            logger.warning(
                "apply_patches: no existing memory for user_id=%s, starting from empty",
                user_id,
            )
            current = {}

        # Merge with the skeleton so intermediate paths always exist.
        # Existing values win; skeleton only fills in missing keys.
        import copy

        base = copy.deepcopy(_PERSONA_SKELETON)
        for key, val in current.items():
            base[key] = val
        current = base

        updated = _safe_apply_patches(current, patches)
        logger.info(
            "apply_patches: applied %d operations to user_id=%s",
            len(patches),
            user_id,
        )

        return await self.upsert_user_memory(user_id, updated)

    async def ensure_user_memory(self, user_id: str, default: dict) -> dict:
        """Return existing memory or initialise with *default* if absent."""
        existing = await self.get_user_memory(user_id)
        if existing:
            return existing
        logger.info("ensure_user_memory: initialising memory for user_id=%s", user_id)
        return await self.upsert_user_memory(user_id, default)
