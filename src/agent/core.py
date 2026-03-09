"""
Multi-agent pipeline for Jarvis.

Pipeline
--------
START
  └── gatekeeper_node   (fetch user memory + LLM decision)
        ├── [trigger=True]  → extractor_node  (extract facts & update SurrealDB)
        │                         └── core_agent_node  (conversation)
        └── [trigger=False] → core_agent_node (conversation)
                                    └── END
"""

import json
import logging
import os
import re
from enum import Enum
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agent.memory.conversation_store import ConversationStore
from src.agent.memory.store import MemoryStore
from src.config import OPENAI_API_KEY, OPENAI_BASE_URL

from .mcp import MCPClientManager

logger = logging.getLogger("agent")


LLM_MODEL=os.getenv("LLM_MODEL", "deepseek-chat")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompt(agent_name: str, template_name: str, **kwargs) -> str:
    """Load a prompt template and substitute {{variable}} placeholders.

    Uses ``{{variable}}`` (double-brace) syntax so that literal JSON braces
    inside the templates are never accidentally consumed by Python's
    ``str.format()``.
    """
    prompt_path = os.path.join(os.path.dirname(__file__), agent_name, template_name)
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()
        for key, value in kwargs.items():
            template = template.replace("{{" + key + "}}", str(value))
        return template
    except Exception as exc:
        logger.error("Error loading prompt %s: %s", template_name, exc)
        return "You are a helpful AI assistant."


def _load_file(path: str) -> str:
    """Read a file to string; return empty string on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _parse_json_from_llm(text: str) -> dict | list | None:
    """Strip markdown code fences and parse the first JSON object or array."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip("`").strip()
    for opener, closer in [('[', ']'), ('{', '}')]:
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break
    return None


# ---------------------------------------------------------------------------
# MCP client manager
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class AgentStatus(AttributeError, Enum):
    """Indicates the current status of the agent's processing pipeline."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    RESPONDING = "responding"
    COMPLETE = "complete"
    ERROR = "error"

class AgentState(TypedDict):
    # Persistent conversation history — add_messages APPENDS each update
    messages: Annotated[list, add_messages]
    # Per-turn context
    user_id: str
    user_name: str
    user_input: str
    platform: str
    # Set by gatekeeper_node
    gatekeeper_result: dict | None
    # Fetched/updated across nodes; NOT stored in the LangGraph checkpointer
    # (SurrealDB is the source of truth for long-term memory)
    user_memory: dict | None
    # Short-term memory: SurrealDB conversation session
    conversation_id: str | None        # str form of RecordID
    persisted_history: list | None     # recent BaseMessage objects loaded from SurrealDB


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AgentWithWorkflow:
    """LangGraph-based multi-agent pipeline with SurrealDB-backed long-term memory."""

    # Proto schema text — loaded once at class level for the extractor prompt
    _PROTO_SCHEMA: str = _load_file(
        os.path.join(os.path.dirname(__file__), "memory", "persona.proto")
    )

    def __init__(self, agent_id: str = 'jarvis') -> None:
        self.agent_id = agent_id
        self.memory_saver = MemorySaver()
        self.memory_store = MemoryStore()
        self.conversation_store = ConversationStore()
        self.mcp_client = MCPClientManager()

        # Conversational LLM (slightly creative)
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY or "dummy_key_if_not_set",
            base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None,
            model=LLM_MODEL,
            temperature=0.7,
        )
        # Structured extraction / classification (deterministic)
        self.llm_precise = ChatOpenAI(
            api_key=OPENAI_API_KEY or "dummy_key_if_not_set",
            base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None,
            model=LLM_MODEL,
            temperature=0.0,
        )

        # ------------------------------------------------------------------
        # Build the state graph
        # ------------------------------------------------------------------
        graph = StateGraph(AgentState)
        graph.add_node("gatekeeper", self._gatekeeper_node)
        graph.add_node("extractor", self._extractor_node)
        graph.add_node("core_agent", self._core_agent_node)

        graph.add_edge(START, "gatekeeper")
        graph.add_conditional_edges(
            "gatekeeper",
            self._route_after_gatekeeper,
            {"extract": "extractor", "respond": "core_agent"},
        )
        graph.add_edge("extractor", "core_agent")
        graph.add_edge("core_agent", END)

        self.app = graph.compile(checkpointer=self.memory_saver)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_after_gatekeeper(
        self, state: AgentState
    ) -> Literal["extract", "respond"]:
        result = state.get("gatekeeper_result") or {}
        if result.get("trigger") is True:
            logger.info(
                "Gatekeeper triggered memory update | category=%s | %s",
                result.get("category"),
                result.get("reasoning"),
            )
            return "extract"
        return "respond"

    # ------------------------------------------------------------------
    # Node: Gatekeeper
    # ------------------------------------------------------------------

    async def _gatekeeper_node(self, state: AgentState) -> dict:
        """Fetch user memory from SurrealDB and decide if this turn needs a memory update."""
        user_id = state["user_id"]
        user_input = state["user_input"]
        platform = state.get("platform", "discord")

        # 1. Fetch current user memory from SurrealDB.
        #    If this is the user's first interaction, bootstrap a record with
        #    the skeleton + their display name so the extractor always has a
        #    non-empty base to patch into.
        user_name = state.get("user_name", "")
        try:
            await self.memory_store.connect()
            user_memory = await self.memory_store.get_user_memory(user_id)
            if user_memory is None:
                import copy

                from src.agent.memory.store import _PERSONA_SKELETON
                seed = copy.deepcopy(_PERSONA_SKELETON)
                seed["demographics"]["preferred_name"] = user_name
                user_memory = await self.memory_store.upsert_user_memory(user_id, seed)
                logger.info(
                    "gatekeeper: bootstrapped new memory record for user_id=%s name=%r",
                    user_id, user_name,
                )
        except Exception as exc:
            logger.error("gatekeeper: memory fetch failed: %s", exc)
            user_memory = None

        # 2. Ensure the conversation session exists and load recent history from SurrealDB.
        conversation_id = None
        persisted_history: list[BaseMessage] = []
        try:
            await self.conversation_store.connect()
            conversation_id = await self.conversation_store.get_or_create_conversation(
                user_id, platform, agent_id="jarvis"
            )
            persisted_history = await self.conversation_store.load_as_langchain_messages(
                conversation_id, limit=50
            )
            logger.debug(
                "gatekeeper: loaded %d messages from conversation %s",
                len(persisted_history),
                conversation_id,
            )
        except Exception as exc:
            logger.warning("gatekeeper: conversation fetch failed: %s", exc)

        # 3. Build a brief conversation context for the gatekeeper
        # Prefer persisted history; fall back to LangGraph in-memory state.
        context_messages = persisted_history[-6:] if persisted_history else list(state.get("messages", []))[-6:]
        lines = []
        for m in context_messages:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            lines.append(f"{role}: {m.content}")
        lines.append(f"User: {user_input}")
        gatekeeper_input = "\n".join(lines)

        prompt = load_prompt(
            "memory", "prompts/gatekeeper.md", input=gatekeeper_input
        )

        # 3. Call the gatekeeper LLM
        try:
            response = await self.llm_precise.ainvoke(
                [HumanMessage(content=prompt)]
            )
            gatekeeper_result = _parse_json_from_llm(response.content)
            if not isinstance(gatekeeper_result, dict):
                logger.warning(
                    "gatekeeper: unparseable response: %s", response.content[:300]
                )
                gatekeeper_result = {
                    "trigger": False,
                    "category": None,
                    "reasoning": "parse error",
                }
        except Exception as exc:
            logger.error("gatekeeper LLM call failed: %s", exc)
            gatekeeper_result = {
                "trigger": False,
                "category": None,
                "reasoning": str(exc),
            }

        return {
            "user_memory": user_memory,
            "gatekeeper_result": gatekeeper_result,
            "conversation_id": str(conversation_id) if conversation_id else None,
            "persisted_history": persisted_history,
        }

    # ------------------------------------------------------------------
    # Node: Extractor
    # ------------------------------------------------------------------

    async def _extractor_node(self, state: AgentState) -> dict:
        """Extract structured facts from the user's input and persist them to SurrealDB."""
        user_id = state["user_id"]
        user_input = state["user_input"]
        current_memory = state.get("user_memory") or {}

        prompt = load_prompt(
            "memory",
            "prompts/extractor.md",
            PROTO_SCHEMA_DOC=self._PROTO_SCHEMA,
            CURRENT_USER_JSON=json.dumps(current_memory, ensure_ascii=False, indent=2),
            new_input=user_input,
        )

        # 1. Call the extractor LLM
        try:
            response = await self.llm_precise.ainvoke(
                [HumanMessage(content=prompt)]
            )
            patches = _parse_json_from_llm(response.content)
            # Only accept a list of patch operations; a bare object is invalid.
            if not isinstance(patches, list):
                logger.warning(
                    "extractor: unexpected patch format: %s", response.content[:300]
                )
                patches = []
        except Exception as exc:
            logger.error("extractor LLM call failed: %s", exc)
            patches = []

        # 2. Apply JSON patches to SurrealDB
        if patches:
            try:
                updated_memory = await self.memory_store.apply_patches(
                    user_id, patches
                )
                logger.info(
                    "extractor: applied %d patches for user_id=%s",
                    len(patches),
                    user_id,
                )
            except Exception as exc:
                logger.error(
                    "extractor: patch apply failed for user_id=%s: %s", user_id, exc
                )
                updated_memory = current_memory
        else:
            logger.info("extractor: no patches to apply for user_id=%s", user_id)
            updated_memory = current_memory

        return {"user_memory": updated_memory}

    # ------------------------------------------------------------------
    # Node: Core Agent
    # ------------------------------------------------------------------

    async def _core_agent_node(self, state: AgentState) -> dict:
        """Main conversational node: builds a memory-aware system prompt and responds.

        Supports tool-calling via the local MCP server — the LLM may invoke any
        tool exposed by the MCP server before producing a final reply.
        """
        user_name = state["user_name"]
        user_memory = state.get("user_memory") or {}

        # Build a fresh system prompt every turn with the latest memory snapshot.
        soul_prompt = load_prompt("jarvis", "soul.md")
        memory_json = (
            json.dumps(user_memory, ensure_ascii=False, indent=2)
            if user_memory
            else "No memory recorded yet."
        )
        sys_content = load_prompt(
            "jarvis",
            "system_prompt.md",
            user_name=user_name,
            agent_soul=soul_prompt,
            user_memory=memory_json,
        )

        # Full message list for the LLM:
        # [system] + past history (without current turn) + current HumanMessage
        #
        # When persisted_history is available it does NOT include the current
        # user message (it was loaded before this turn), so we always append
        # the current user_input explicitly.
        # When falling back to LangGraph in-memory state, add_messages has
        # already appended the current HumanMessage as the last item — so we
        # drop the last element and re-add it manually for consistency.
        user_input = state.get("user_input", "")
        persisted = state.get("persisted_history") or []
        if persisted:
            history = persisted  # does not contain the current message
        else:
            state_msgs = list(state.get("messages", []))
            # Drop the trailing HumanMessage that add_messages injected so we
            # can append it uniformly below.
            history = state_msgs[:-1] if state_msgs else []

        logger.info("core_agent: history=%d messages for user_input=%r", len(history), user_input[:60])
        messages_for_llm: list[BaseMessage] = (
            [SystemMessage(content=sys_content)]
            + history
            + [HumanMessage(content=user_input)]
        )

        # Load MCP tools and bind them to the LLM (cached after first call).
        tools = await self.mcp_client.get_tools()
        llm_with_tools = self.llm.bind_tools(tools) if tools else self.llm
        tool_map: dict[str, object] = {t.name: t for t in tools}

        # ------------------------------------------------------------------
        # Tool-calling loop (ReAct style)
        # The LLM may call one or more tools before producing a final answer.
        # We loop until the model returns a message with no tool_calls.
        # ------------------------------------------------------------------
        response: AIMessage = AIMessage(content="")  # ensure always bound
        for _ in range(10):  # guard against infinite loops
            response = await llm_with_tools.ainvoke(messages_for_llm)
            if not getattr(response, "tool_calls", None):
                break  # Final answer — no more tool calls requested

            # Execute each requested tool call and append the results.
            messages_for_llm.append(response)
            for tc in response.tool_calls:
                tool = tool_map.get(tc["name"])
                if tool is None:
                    tool_result = f"Unknown tool: {tc['name']}"
                    logger.warning("core_agent: unknown tool '%s' requested", tc["name"])
                else:
                    try:
                        tool_result = await tool.ainvoke(tc["args"])
                        logger.debug(
                            "core_agent: tool '%s' returned: %s", tc["name"], str(tool_result)[:200]
                        )
                    except Exception as exc:
                        tool_result = f"Tool error: {exc}"
                        logger.error(
                            "core_agent: tool '%s' raised: %s", tc["name"], exc
                        )
                messages_for_llm.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tc["id"])
                )
        else:
            # Loop exhausted without a clean break — force a plain text reply.
            logger.warning("core_agent: tool loop exhausted (10 iterations); forcing final text response")
            response = await self.llm.ainvoke(messages_for_llm)

        # Guard against models that return empty content (e.g. some tool-call-only responses).
        if not response.content:
            logger.warning("core_agent: LLM returned empty content; retrying without tools")
            response = await self.llm.ainvoke(messages_for_llm)

        # Persist this turn (user message + assistant reply) to SurrealDB.
        conversation_id_str = state.get("conversation_id")
        user_input = state.get("user_input", "")
        if conversation_id_str:
            try:
                from surrealdb import RecordID as _RID
                # RecordID can be reconstructed from its string representation.
                conv_id = _RID(*conversation_id_str.split(":", 1)) if ":" in conversation_id_str else conversation_id_str
                await self.conversation_store.append_message(conv_id, "user", user_input)
                await self.conversation_store.append_message(
                    conv_id, "assistant", response.content
                )
                logger.debug(
                    "core_agent: persisted turn to conversation %s", conv_id
                )
            except Exception as exc:
                logger.warning(
                    "core_agent: failed to persist conversation turn: %s", exc
                )

        return {"messages": [response]}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        user_id: str,
        user_name: str,
        user_input: str,
        platform: str = "discord",
    ) -> str:
        """Run the full multi-agent pipeline for one user turn and return the reply."""
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_or_deepseek_api_key_here":
            return f"⚠️ API key not configured yet.\n\nYour message: {user_input}"

        config = {"configurable": {"thread_id": str(user_id)}}

        try:
            final_state = await self.app.ainvoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "user_id": user_id,
                    "user_name": user_name,
                    "user_input": user_input,
                    "platform": platform,
                    "gatekeeper_result": None,
                    "user_memory": None,
                    "conversation_id": None,
                    "persisted_history": None,
                },
                config,
            )
            return final_state["messages"][-1].content
        except Exception as exc:
            logger.error("chat: pipeline error: %s", exc)
            return f"⚠️ I encountered an error while processing your request: {exc}"
