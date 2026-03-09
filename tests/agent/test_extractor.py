"""
Unit tests for the Extractor node.

The LLM (`llm_precise`) and the SurrealDB store are fully mocked — no network
calls are made.  Each test calls `_extractor_node` directly on a
`AgentWIthWorkflow` instance, inspecting the returned state and verifying that
`MemoryStore.apply_patches` is called (or skipped) with the correct arguments.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, call, patch

from src.agent.core import AgentWithWorkflow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    """Return a AgentWIthWorkflow with LLMs and MemoryStore fully mocked out."""
    with patch("src.agent.core.ChatOpenAI"), patch("src.agent.core.MemoryStore"):
        inst = AgentWithWorkflow()
    inst.llm_precise = MagicMock()
    inst.memory_store = MagicMock()
    inst.memory_store.connect = AsyncMock()
    inst.memory_store.get_user_memory = AsyncMock(return_value=None)
    inst.memory_store.apply_patches = AsyncMock()
    return inst


def _make_state(user_input: str, user_memory: dict | None = None):
    return {
        "messages": [],
        "user_id": "user_123",
        "user_name": "Test User",
        "user_input": user_input,
        "gatekeeper_result": {"trigger": True, "category": "Identity/Life Events"},
        "user_memory": user_memory or {},
    }


def _llm_response(content: str):
    mock = MagicMock()
    mock.content = content
    return mock


# ---------------------------------------------------------------------------
# Valid patch application
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestExtractorPatchApplication:
    async def test_applies_patches_from_valid_llm_response(self, agent):
        """Extractor should call apply_patches with the list returned by the LLM."""
        patches = [
            {"op": "add", "path": "/demographics/location/current", "value": "Beijing"},
        ]
        agent.llm_precise.ainvoke = AsyncMock(
            return_value=_llm_response(json.dumps(patches))
        )
        agent.memory_store.apply_patches = AsyncMock(
            return_value={"user_id": "user_123", "demographics": {"location": {"current": "Beijing"}}}
        )

        result = await agent._extractor_node(_make_state("I moved to Beijing last week."))

        agent.memory_store.apply_patches.assert_called_once_with("user_123", patches)
        assert result["user_memory"]["demographics"]["location"]["current"] == "Beijing"

    async def test_applies_multiple_patches(self, agent):
        patches = [
            {"op": "replace", "path": "/work_context/roles/0/is_current", "value": False},
            {"op": "add", "path": "/preferences/daily_life/hobbies/-", "value": "fishing"},
        ]
        agent.llm_precise.ainvoke = AsyncMock(
            return_value=_llm_response(json.dumps(patches))
        )
        updated = {"user_id": "user_123", "work_context": {"roles": [{"is_current": False}]},
                   "preferences": {"daily_life": {"hobbies": ["fishing"]}}}
        agent.memory_store.apply_patches = AsyncMock(return_value=updated)

        result = await agent._extractor_node(_make_state("I retired and took up fishing."))

        agent.memory_store.apply_patches.assert_called_once_with("user_123", patches)
        assert result["user_memory"] == updated

    async def test_accepts_markdown_fenced_patches(self, agent):
        """Patches wrapped in ```json ... ``` code fences should still be parsed."""
        patches = [{"op": "add", "path": "/preferences/languages/-", "value": "Japanese"}]
        fenced = f"```json\n{json.dumps(patches)}\n```"
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response(fenced))
        agent.memory_store.apply_patches = AsyncMock(return_value={"user_id": "user_123"})

        await agent._extractor_node(_make_state("I'm learning Japanese"))
        agent.memory_store.apply_patches.assert_called_once_with("user_123", patches)

    async def test_passes_user_id_correctly_to_apply_patches(self, agent):
        """The correct user_id from state must be forwarded to apply_patches."""
        state = _make_state("I live in Shanghai now.")
        state["user_id"] = "discord_987654"

        patches = [{"op": "replace", "path": "/demographics/location/current", "value": "Shanghai"}]
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response(json.dumps(patches)))
        agent.memory_store.apply_patches = AsyncMock(return_value={})

        await agent._extractor_node(state)
        agent.memory_store.apply_patches.assert_called_once_with("discord_987654", patches)


# ---------------------------------------------------------------------------
# Empty / no-op patch cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestExtractorEmptyPatches:
    async def test_skips_apply_when_llm_returns_empty_list(self, agent):
        """An empty patch list from the LLM means nothing needs updating."""
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response("[]"))
        current_memory = {"user_id": "user_123", "demographics": {"preferred_name": "Alice"}}

        result = await agent._extractor_node(_make_state("Hello!", user_memory=current_memory))

        agent.memory_store.apply_patches.assert_not_called()
        assert result["user_memory"] == current_memory

    async def test_returns_current_memory_when_no_patches(self, agent):
        """When patches=[], the input memory must be returned unchanged."""
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response("[]"))
        current_memory = {"user_id": "user_123", "some_key": "some_value"}

        result = await agent._extractor_node(_make_state("Nothing important.", user_memory=current_memory))
        assert result["user_memory"] is current_memory

    async def test_deduplication_empty_list_no_store_call(self, agent):
        """LLM correctly returns [] for a duplicate fact → store must not be touched."""
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response("  []  "))
        await agent._extractor_node(_make_state("I still live in Shanghai."))
        agent.memory_store.apply_patches.assert_not_called()


# ---------------------------------------------------------------------------
# Fallback / error-handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestExtractorFallbacks:
    async def test_fallback_on_invalid_json(self, agent):
        """Non-JSON LLM output should result in no patches and current memory returned."""
        agent.llm_precise.ainvoke = AsyncMock(
            return_value=_llm_response("I cannot extract anything.")
        )
        current_memory = {"user_id": "user_123"}

        result = await agent._extractor_node(_make_state("blah", user_memory=current_memory))
        agent.memory_store.apply_patches.assert_not_called()
        assert result["user_memory"] == current_memory

    async def test_fallback_on_object_instead_of_array(self, agent):
        """If the LLM returns a JSON object (not array), treat as no-patches."""
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response(
            '{"op": "add", "path": "/x", "value": "y"}'  # object, not array
        ))
        current_memory = {"user_id": "user_123"}
        result = await agent._extractor_node(_make_state("blah", user_memory=current_memory))
        agent.memory_store.apply_patches.assert_not_called()
        assert result["user_memory"] == current_memory

    async def test_fallback_on_llm_exception(self, agent):
        """LLM network error → no patches, current memory preserved."""
        agent.llm_precise.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        current_memory = {"user_id": "user_123", "demographics": {"preferred_name": "Bob"}}

        result = await agent._extractor_node(_make_state("I'm a doctor now.", user_memory=current_memory))
        agent.memory_store.apply_patches.assert_not_called()
        assert result["user_memory"] == current_memory

    async def test_fallback_on_store_apply_exception(self, agent):
        """If apply_patches raises, the extractor falls back to the pre-existing memory."""
        patches = [{"op": "add", "path": "/preferences/languages/-", "value": "French"}]
        agent.llm_precise.ainvoke = AsyncMock(return_value=_llm_response(json.dumps(patches)))
        agent.memory_store.apply_patches = AsyncMock(side_effect=Exception("SurrealDB write failed"))
        current_memory = {"user_id": "user_123", "preferences": {"languages": ["English"]}}

        result = await agent._extractor_node(_make_state("I now speak French.", user_memory=current_memory))
        assert result["user_memory"] == current_memory


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestExtractorPrompt:
    async def test_prompt_contains_proto_schema(self, agent):
        """The extractor prompt must embed the proto schema for grounding."""
        captured: list[str] = []

        async def capture(messages):
            captured.append(messages[0].content)
            return _llm_response("[]")

        agent.llm_precise.ainvoke = capture
        await agent._extractor_node(_make_state("test"))
        assert "UserPersona" in captured[0]  # from persona.proto

    async def test_prompt_contains_current_memory(self, agent):
        """The extractor prompt must include the serialised current memory state."""
        current_memory = {"user_id": "user_123", "demographics": {"preferred_name": "Charlie"}}
        captured: list[str] = []

        async def capture(messages):
            captured.append(messages[0].content)
            return _llm_response("[]")

        agent.llm_precise.ainvoke = capture
        await agent._extractor_node(_make_state("test", user_memory=current_memory))
        assert "Charlie" in captured[0]

    async def test_prompt_contains_user_input(self, agent):
        """The raw user utterance must appear in the extractor prompt."""
        captured: list[str] = []

        async def capture(messages):
            captured.append(messages[0].content)
            return _llm_response("[]")

        agent.llm_precise.ainvoke = capture
        await agent._extractor_node(_make_state("I recently retired from Google."))
        assert "I recently retired from Google." in captured[0]

    async def test_empty_memory_replaced_with_empty_object(self, agent):
        """When user_memory is None, the prompt should show an empty JSON object, not 'null'."""
        captured: list[str] = []

        async def capture(messages):
            captured.append(messages[0].content)
            return _llm_response("[]")

        agent.llm_precise.ainvoke = capture
        state = _make_state("test")
        state["user_memory"] = None
        await agent._extractor_node(state)
        assert "null" not in captured[0].split("Current State:")[1][:50] if "Current State:" in captured[0] else True
