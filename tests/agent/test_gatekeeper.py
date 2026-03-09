"""
Unit tests for the Gatekeeper node and routing logic.

The LLM (`llm_precise`) and the SurrealDB store are fully mocked — no network
calls are made.  Each test calls `_gatekeeper_node` or `_route_after_gatekeeper`
directly on a `AgentWIthWorkflow` instance.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from src.agent.core import AgentWithWorkflow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    """Return a AgentWIthWorkflow with LLMs and MemoryStore fully mocked out."""
    with patch("src.agent.core.ChatOpenAI"), patch("src.agent.core.MemoryStore"):
        inst = AgentWithWorkflow()
    # Replace with controllable mocks
    inst.llm_precise = MagicMock()
    inst.memory_store = MagicMock()
    inst.memory_store.connect = AsyncMock()
    inst.memory_store.get_user_memory = AsyncMock(return_value=None)
    return inst


def _make_state(user_input: str, messages: list | None = None, user_memory=None):
    return {
        "messages": messages or [],
        "user_id": "user_123",
        "user_name": "Test User",
        "user_input": user_input,
        "gatekeeper_result": None,
        "user_memory": user_memory,
    }


def _llm_response(content: str):
    """Wrap a string as an AIMessage returned by `ainvoke`."""
    mock = MagicMock()
    mock.content = content
    return mock


# ---------------------------------------------------------------------------
# Routing tests (pure sync, no I/O)
# ---------------------------------------------------------------------------

class TestRouting:
    def test_route_extract_when_triggered(self, agent):
        state = _make_state("")
        state["gatekeeper_result"] = {"trigger": True, "category": "Identity/Life Events"}
        assert agent._route_after_gatekeeper(state) == "extract"

    def test_route_respond_when_not_triggered(self, agent):
        state = _make_state("")
        state["gatekeeper_result"] = {"trigger": False, "category": None}
        assert agent._route_after_gatekeeper(state) == "respond"

    def test_route_respond_on_none_gatekeeper_result(self, agent):
        state = _make_state("")
        state["gatekeeper_result"] = None
        assert agent._route_after_gatekeeper(state) == "respond"

    def test_route_respond_on_empty_dict(self, agent):
        state = _make_state("")
        state["gatekeeper_result"] = {}
        assert agent._route_after_gatekeeper(state) == "respond"

    def test_route_extract_ignores_category_value(self, agent):
        """trigger=True should route to extract regardless of category content."""
        state = _make_state("")
        state["gatekeeper_result"] = {"trigger": True, "category": None, "reasoning": "x"}
        assert agent._route_after_gatekeeper(state) == "extract"


# ---------------------------------------------------------------------------
# Gatekeeper node — trigger=True cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGatekeeperTriggered:
    async def test_triggers_on_identity_fact(self, agent):
        """Moving to a new city is a persistent life event → trigger=True."""
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response(
            '{"trigger": true, "category": "Identity/Life Events", "reasoning": "User mentioned relocation"}'
        ))
        result = await agent._gatekeeper_node(
            _make_state("老贾，我上周搬到北京了")
        )
        assert result["gatekeeper_result"]["trigger"] is True
        assert result["gatekeeper_result"]["category"] == "Identity/Life Events"

    async def test_triggers_on_job_change(self, agent):
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response(
            '{"trigger": true, "category": "Identity/Life Events", "reasoning": "New job title disclosed"}'
        ))
        result = await agent._gatekeeper_node(
            _make_state("I just became CTO at a new startup.")
        )
        assert result["gatekeeper_result"]["trigger"] is True

    async def test_triggers_on_explicit_instruction(self, agent):
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response(
            '{"trigger": true, "category": "Explicit Instructions", "reasoning": "User gave a standing order"}'
        ))
        result = await agent._gatekeeper_node(
            _make_state("Always reply to me in English, never Chinese.")
        )
        assert result["gatekeeper_result"]["trigger"] is True
        assert result["gatekeeper_result"]["category"] == "Explicit Instructions"

    async def test_triggers_on_long_term_goal(self, agent):
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response(
            '{"trigger": true, "category": "Long-term Goals", "reasoning": "User stated a goal"}'
        ))
        result = await agent._gatekeeper_node(
            _make_state("I'm trying to learn Japanese this year.")
        )
        assert result["gatekeeper_result"]["trigger"] is True

    async def test_triggers_on_preference(self, agent):
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response(
            '{"trigger": true, "category": "Preferences", "reasoning": "Dietary preference stated"}'
        ))
        result = await agent._gatekeeper_node(
            _make_state("I became vegan three months ago.")
        )
        assert result["gatekeeper_result"]["trigger"] is True


# ---------------------------------------------------------------------------
# Gatekeeper node — trigger=False cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGatekeeperNotTriggered:
    async def test_no_trigger_on_chit_chat(self, agent):
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response(
            '{"trigger": false, "category": null, "reasoning": "Routine greeting"}'
        ))
        result = await agent._gatekeeper_node(_make_state("How are you today?"))
        assert result["gatekeeper_result"]["trigger"] is False
        assert result["gatekeeper_result"]["category"] is None

    async def test_no_trigger_on_temporary_state(self, agent):
        """Transient states like hunger should not be persisted."""
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response(
            '{"trigger": false, "category": null, "reasoning": "Temporary context"}'
        ))
        result = await agent._gatekeeper_node(_make_state("I'm really hungry right now."))
        assert result["gatekeeper_result"]["trigger"] is False

    async def test_no_trigger_on_emotional_outburst(self, agent):
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response(
            '{"trigger": false, "category": null, "reasoning": "Emotional outburst, not a persistent fact"}'
        ))
        result = await agent._gatekeeper_node(_make_state("Ugh, I'm so frustrated today!"))
        assert result["gatekeeper_result"]["trigger"] is False

    async def test_no_trigger_on_meta_talk_about_ai(self, agent):
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response(
            '{"trigger": false, "category": null, "reasoning": "Meta-talk about the AI"}'
        ))
        result = await agent._gatekeeper_node(_make_state("You answered really fast today."))
        assert result["gatekeeper_result"]["trigger"] is False


# ---------------------------------------------------------------------------
# Gatekeeper node — fallback / error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGatekeeperFallbacks:
    async def test_fallback_on_unparseable_llm_response(self, agent):
        """If the LLM returns something that isn't JSON, default to trigger=False."""
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response(
            "Sorry, I cannot decide right now."  # plain text, no JSON
        ))
        result = await agent._gatekeeper_node(_make_state("I live in Shanghai."))
        assert result["gatekeeper_result"]["trigger"] is False
        assert result["gatekeeper_result"]["reasoning"] == "parse error"

    async def test_fallback_on_llm_exception(self, agent):
        """If the LLM call itself errors, default to trigger=False."""
        agent.llm_precise.ainvoke = AsyncMock(side_effect=RuntimeError("API timeout"))
        result = await agent._gatekeeper_node(_make_state("I work at Google."))
        assert result["gatekeeper_result"]["trigger"] is False
        assert "API timeout" in result["gatekeeper_result"]["reasoning"]

    async def test_memory_fetch_failure_graceful(self, agent):
        """If SurrealDB is unreachable, user_memory=None and pipeline continues normally."""
        agent.memory_store.connect = AsyncMock(side_effect=ConnectionError("DB unreachable"))
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response(
            '{"trigger": false, "category": null, "reasoning": "chit-chat"}'
        ))
        result = await agent._gatekeeper_node(_make_state("Hello!"))
        assert result["user_memory"] is None
        assert result["gatekeeper_result"]["trigger"] is False

    async def test_memory_returned_in_state(self, agent):
        """The fetched user memory should be passed through in the returned state."""
        stored_memory = {"user_id": "user_123", "demographics": {"preferred_name": "Alice"}}
        agent.memory_store.get_user_memory = AsyncMock(return_value=stored_memory)
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response(
            '{"trigger": false, "category": null, "reasoning": "chit-chat"}'
        ))
        result = await agent._gatekeeper_node(_make_state("Hi there!"))
        assert result["user_memory"] == stored_memory


# ---------------------------------------------------------------------------
# Gatekeeper node — prompt construction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGatekeeperPrompt:
    async def test_prompt_includes_user_input(self, agent):
        """The gatekeeper prompt must contain the current user_input."""
        captured: list[str] = []

        async def capture_invoke(messages):
            captured.append(messages[0].content)
            return _llm_response('{"trigger": false, "category": null, "reasoning": "x"}')

        agent.llm_precise.ainvoke = capture_invoke
        await agent._gatekeeper_node(_make_state("I moved to Tokyo recently."))
        assert "I moved to Tokyo recently." in captured[0]

    async def test_prompt_includes_recent_history(self, agent):
        """The last few messages from history must be present in the gatekeeper prompt."""
        history = [
            HumanMessage(content="What's the weather?"),
            AIMessage(content="I don't have real-time data."),
        ]
        captured: list[str] = []

        async def capture_invoke(messages):
            captured.append(messages[0].content)
            return _llm_response('{"trigger": false, "category": null, "reasoning": "x"}')

        agent.llm_precise.ainvoke = capture_invoke
        await agent._gatekeeper_node(_make_state("I prefer Python.", messages=history))
        assert "What's the weather?" in captured[0]
        assert "I don't have real-time data." in captured[0]

    async def test_markdown_fenced_json_is_parsed(self, agent):
        """The gatekeeper should parse JSON even when wrapped in ```json ... ``` fences."""
        fenced = '```json\n{"trigger": true, "category": "Preferences", "reasoning": "ok"}\n```'
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response(fenced))
        result = await agent._gatekeeper_node(_make_state("I prefer dark mode."))
        assert result["gatekeeper_result"]["trigger"] is True
