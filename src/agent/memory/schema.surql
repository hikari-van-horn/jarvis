-- ─────────────────────────────────────────────────────────────────────────────
-- Jarvis Memory Schema
-- Apply with: surreal import --conn ... --user ... --pass ... --ns main --db main schema.surql
-- Or call MemoryStore.init_schema() / ConversationStore.init_schema() at startup.
-- ─────────────────────────────────────────────────────────────────────────────


-- ─────────────────────────────────────────────────────────────────────────────
-- Long-term memory: user persona / facts
-- SCHEMALESS so arbitrary persona fields (demographics, work_context, etc.)
-- are stored without needing individual DEFINE FIELD statements.
-- OVERWRITE ensures this DDL re-applies on every startup / schema migration.
-- ─────────────────────────────────────────────────────────────────────────────

DEFINE TABLE OVERWRITE user_memory SCHEMALESS;

DEFINE FIELD OVERWRITE user_id    ON TABLE user_memory TYPE string;
DEFINE FIELD OVERWRITE updated_at ON TABLE user_memory TYPE datetime
    VALUE time::now();

DEFINE INDEX OVERWRITE idx_user_memory_user_id
    ON TABLE user_memory COLUMNS user_id UNIQUE;


-- ─────────────────────────────────────────────────────────────────────────────
-- Short-term memory: conversation sessions
--
-- One record per chat session between Jarvis and a single user on a platform.
-- A new session is created when the user returns after a configurable idle gap.
-- ─────────────────────────────────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS conversation SCHEMAFULL;

-- Who / where
DEFINE FIELD IF NOT EXISTS agent_id      ON TABLE conversation TYPE string;  -- e.g. "jarvis"
DEFINE FIELD IF NOT EXISTS user_id      ON TABLE conversation TYPE string;
DEFINE FIELD IF NOT EXISTS platform     ON TABLE conversation TYPE string;  -- "discord" | "telegram" | ...

-- Lifecycle timestamps
DEFINE FIELD IF NOT EXISTS started_at     ON TABLE conversation TYPE datetime
    VALUE $before OR time::now();
DEFINE FIELD IF NOT EXISTS last_active_at ON TABLE conversation TYPE datetime
    VALUE time::now();

-- Optional free-form bag for future use (channel id, guild id, etc.)
DEFINE FIELD IF NOT EXISTS metadata ON TABLE conversation
    TYPE option<object>;

-- Fast lookups
DEFINE INDEX IF NOT EXISTS idx_conv_user
    ON TABLE conversation COLUMNS user_id;
DEFINE INDEX IF NOT EXISTS idx_conv_user_platform
    ON TABLE conversation COLUMNS user_id, platform;
DEFINE INDEX IF NOT EXISTS idx_conv_agent_user_platform
    ON TABLE conversation COLUMNS agent_id, user_id, platform;


-- ─────────────────────────────────────────────────────────────────────────────
-- Short-term memory: individual messages inside a conversation
--
-- Each row is one conversational turn.  Ordered by created_at.
-- ─────────────────────────────────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS message SCHEMAFULL;

-- Parent session (typed record link)
DEFINE FIELD IF NOT EXISTS conversation ON TABLE message TYPE record<conversation>;

-- Denormalised for faster per-user queries without joining
DEFINE FIELD IF NOT EXISTS user_id  ON TABLE message TYPE string;

-- "user" | "assistant" | "system"
DEFINE FIELD IF NOT EXISTS role ON TABLE message TYPE string
    ASSERT $value INSIDE ["user", "assistant", "system"];

DEFINE FIELD IF NOT EXISTS content    ON TABLE message TYPE string;

DEFINE FIELD IF NOT EXISTS created_at ON TABLE message TYPE datetime
    VALUE $before OR time::now();

-- Optional metadata: model name, token counts, latency, etc.
DEFINE FIELD IF NOT EXISTS metadata ON TABLE message
    TYPE option<object>;

-- Retrieve all messages for a conversation, in order
DEFINE INDEX IF NOT EXISTS idx_msg_conversation_created
    ON TABLE message COLUMNS conversation, created_at;

-- Retrieve all messages for a user across all sessions
DEFINE INDEX IF NOT EXISTS idx_msg_user
    ON TABLE message COLUMNS user_id;
