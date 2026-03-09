import pytest

from src.agent.core import load_prompt


def test_load_prompt():
    """Test that the load_prompt function resolves the markdown file and formats it properly."""
    try:
        soul_prompt = load_prompt("prompts/jarvis/soul.md")
        prompt = load_prompt("prompts/jarvis/system.md", user_name="TestUser", agent_soul=soul_prompt)
        assert "TestUser" in prompt
        assert "Jarvis" in prompt
    except Exception as e:
        pytest.fail(f"Failed to load prompt: {e}")
