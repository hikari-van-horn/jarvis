# Jarvis Agent — Architecture

## Pipeline

```
START
  └── gatekeeper_node  ──→ [trigger=True]  → extractor_node → core_agent_node → END
                       ──→ [trigger=False]                  → core_agent_node → END
```

---

## Nodes

| Node | Role |
|---|---|
| `_gatekeeper_node` | Fetches `user_memory` from SurrealDB, runs gatekeeper LLM with last 3 turns of context, returns `{"trigger": bool, ...}` |
| `_extractor_node` | (conditional) Calls extractor LLM with proto schema + current memory, produces RFC 6902 JSON Patch, applies it to SurrealDB |
| `_core_agent_node` | Builds a **fresh** system prompt each turn (soul + latest memory), invokes conversational LLM over the full persistent history |

The system prompt is **never stored** in the LangGraph checkpointer — SurrealDB is the source of truth for long-term memory.

---

## Files

### `core.py`
Full multi-agent LangGraph pipeline. `AgentWIthWorkflow` exposes a single async method:
```python
await agent.chat(user_id, user_name, user_input) -> str
```

### `memory/store.py`
`MemoryStore` wraps `AsyncSurreal` (SurrealDB HTTP client):

| Method | Description |
|---|---|
| `connect()` | Authenticate and select namespace/db (idempotent) |
| `get_user_memory(user_id)` | Fetch persona document |
| `upsert_user_memory(user_id, data)` | Full replace |
| `apply_patches(user_id, patches)` | Apply RFC 6902 JSON Patch list, then upsert |
| `ensure_user_memory(user_id, default)` | Init from default if no record exists |

### `memory/persona.proto`
Proto3 schema defining the `UserPersona` structure used by the extractor.

### `memory/prompts/gatekeeper.md`
System prompt for the gatekeeper LLM. Classifies whether a user turn contains persistent facts.
Placeholder: `{{input}}`

### `memory/prompts/extractor.md`
System prompt for the extractor LLM. Takes proto schema + current memory state + new input, outputs an RFC 6902 JSON Patch list.
Placeholders: `{{PROTO_SCHEMA_DOC}}`, `{{CURRENT_USER_JSON}}`, `{{new_input}}`

### `jarvis/soul.md`
Jarvis's character definition (background, personality, speech style).

### `jarvis/system_prompt.md`
Core agent system prompt template.
Placeholders: `{{user_name}}`, `{{agent_soul}}`, `{{user_memory}}`

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | LLM API key (DeepSeek-compatible) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | LLM base URL |
| `SURREAL_URL` | `https://jarvis-memory-06eaqhiqatvd50k5l2t75ueuuk.aws-aps1.surreal.cloud/` | SurrealDB endpoint |
| `SURREAL_USER` | `root` | SurrealDB username |
| `SURREAL_PASS` | `root` | SurrealDB password |
| `SURREAL_NS` | `jarvis` | SurrealDB namespace |
| `SURREAL_DB` | `memory` | SurrealDB database |

---

## SurrealDB Schema

Records are stored in the `user_memory` table with IDs of the form `user_memory:u_<user_id>`.

```
user_memory
  └── u_<user_id>   →  UserPersona JSON document
```

---

## Dependencies

- `langgraph` — state graph execution
- `langchain-openai` — LLM client
- `surrealdb` — async SurrealDB HTTP client
- `jsonpatch` — RFC 6902 JSON Patch apply
