"""
Unit tests for src/agent/providers.py.

All tests are fully offline — no LLM calls are made.  The ``infra.toml`` file
is mocked via ``tomllib.load`` so the tests are independent of the real file
content on disk.
"""

import textwrap
import tomllib
from unittest.mock import mock_open, patch

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from src.agent.providers import (
    GoogleGenAIProviderConfig,
    RestProviderConfig,
    _expand_env,
    _get_llm_client_from_cfg,
    _parse_provider_config,
    get_llm_client,
    load_providers,
)

# ---------------------------------------------------------------------------
# _expand_env
# ---------------------------------------------------------------------------


class TestExpandEnv:
    def test_replaces_known_variable(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret")
        assert _expand_env("${MY_KEY}") == "secret"

    def test_replaces_multiple_variables(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "8080")
        assert _expand_env("http://${HOST}:${PORT}") == "http://localhost:8080"

    def test_unknown_variable_becomes_empty_string(self, monkeypatch):
        monkeypatch.delenv("UNDEFINED_VAR_XYZ", raising=False)
        assert _expand_env("${UNDEFINED_VAR_XYZ}") == ""

    def test_plain_string_unchanged(self):
        assert _expand_env("http://localhost:11434") == "http://localhost:11434"

    def test_no_placeholder_unchanged(self):
        assert _expand_env("no-placeholder") == "no-placeholder"


# ---------------------------------------------------------------------------
# _parse_provider_config
# ---------------------------------------------------------------------------


class TestParseProviderConfig:
    def test_defaults_to_rest_when_type_missing(self):
        cfg = _parse_provider_config({"base_url": "http://localhost:11434"})
        assert isinstance(cfg, RestProviderConfig)
        assert cfg.base_url == "http://localhost:11434"

    def test_parses_rest_config(self):
        cfg = _parse_provider_config({"type": "rest", "api_key": "sk-x", "base_url": "https://api.x.com/v1"})
        assert isinstance(cfg, RestProviderConfig)
        assert cfg.api_key == "sk-x"
        assert cfg.base_url == "https://api.x.com/v1"

    def test_parses_google_genai_config(self):
        cfg = _parse_provider_config({"type": "google_genai", "api_key": "gk-x"})
        assert isinstance(cfg, GoogleGenAIProviderConfig)
        assert cfg.api_key == "gk-x"

    def test_rest_temperature_default(self):
        cfg = _parse_provider_config({"type": "rest"})
        assert cfg.temperature == 0.7

    def test_google_model_default(self):
        cfg = _parse_provider_config({"type": "google_genai"})
        assert cfg.model == "gemini-2.0-flash"

    def test_custom_temperature_accepted(self):
        cfg = _parse_provider_config({"type": "rest", "temperature": 0.2})
        assert cfg.temperature == 0.2

    def test_invalid_type_raises_validation_error(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _parse_provider_config({"type": "grpc"})

    def test_extra_fields_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _parse_provider_config({"type": "rest", "unknown_field": "value"})

    def test_temperature_above_range_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _parse_provider_config({"type": "rest", "temperature": 3.0})

    def test_temperature_below_range_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _parse_provider_config({"type": "rest", "temperature": -0.1})


# ---------------------------------------------------------------------------
# _build_provider
# ---------------------------------------------------------------------------


class TestBuildProvider:
    def test_rest_returns_chat_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = RestProviderConfig(api_key="${OPENAI_API_KEY}", base_url="https://api.example.com/v1")
        assert isinstance(_get_llm_client_from_cfg(cfg), ChatOpenAI)

    def test_rest_expands_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
        cfg = RestProviderConfig(api_key="direct-key", base_url="${OPENAI_BASE_URL}")
        model = _get_llm_client_from_cfg(cfg)
        assert model.openai_api_base == "https://api.example.com/v1"

    def test_rest_without_api_key_uses_dummy(self):
        cfg = RestProviderConfig(base_url="http://localhost:11434")
        assert isinstance(_get_llm_client_from_cfg(cfg), ChatOpenAI)

    def test_google_genai_returns_google_model(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "gk-test")
        cfg = GoogleGenAIProviderConfig(type="google_genai", api_key="${GOOGLE_API_KEY}")
        assert isinstance(_get_llm_client_from_cfg(cfg), ChatGoogleGenerativeAI)

    def test_google_genai_expands_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "gk-test")
        cfg = GoogleGenAIProviderConfig(type="google_genai", api_key="${GOOGLE_API_KEY}")
        model = _get_llm_client_from_cfg(cfg)
        assert model.google_api_key.get_secret_value() == "gk-test"

    def test_google_genai_without_api_key_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "gk-via-env")
        cfg = GoogleGenAIProviderConfig(type="google_genai")
        assert isinstance(_get_llm_client_from_cfg(cfg), ChatGoogleGenerativeAI)


# ---------------------------------------------------------------------------
# load_providers — integration over the full parsing pipeline
# ---------------------------------------------------------------------------

_FAKE_TOML = textwrap.dedent("""\
    [providers.deepseek]
    type = "rest"
    api_key = "${OPENAI_API_KEY}"
    base_url = "${OPENAI_BASE_URL}"

    [providers.ollama]
    type = "rest"
    base_url = "http://localhost:11434"

    [providers.google]
    type = "google_genai"
    api_key = "${GOOGLE_API_KEY}"

    [tools.mcp]
    server_url = "${MCP_SERVER_URL}"
""").encode()


@pytest.fixture
def fake_toml(monkeypatch):
    """Patch tomllib.load to return a parsed version of _FAKE_TOML and set env vars."""
    parsed = tomllib.loads(_FAKE_TOML.decode())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("GOOGLE_API_KEY", "gk-fake")
    with patch("builtins.open", mock_open(read_data=_FAKE_TOML)), patch("tomllib.load", return_value=parsed):
        yield


class TestLoadProviders:
    def test_returns_all_three_providers(self, fake_toml):
        result = load_providers()
        assert set(result.keys()) == {"deepseek", "ollama", "google"}

    def test_deepseek_is_rest_config(self, fake_toml):
        result = load_providers()
        assert isinstance(result["deepseek"], RestProviderConfig)

    def test_ollama_is_rest_config(self, fake_toml):
        result = load_providers()
        assert isinstance(result["ollama"], RestProviderConfig)

    def test_google_is_google_genai_config(self, fake_toml):
        result = load_providers()
        assert isinstance(result["google"], GoogleGenAIProviderConfig)


# ---------------------------------------------------------------------------
# get_llm_client
# ---------------------------------------------------------------------------


class TestGetLlmClient:
    def test_returns_chat_openai_for_rest_provider(self, fake_toml):
        model = get_llm_client("deepseek")
        assert isinstance(model, ChatOpenAI)

    def test_returns_google_model_for_google_provider(self, fake_toml):
        model = get_llm_client("google")
        assert isinstance(model, ChatGoogleGenerativeAI)

    def test_raises_key_error_for_unknown_provider(self, fake_toml):
        with pytest.raises(KeyError, match="unknown"):
            get_llm_client("unknown")

    def test_kwargs_override_temperature(self, fake_toml):
        model = get_llm_client("deepseek", temperature=0.1)
        assert model.temperature == 0.1

    def test_kwargs_override_model(self, fake_toml):
        model = get_llm_client("deepseek", model="deepseek-reasoner")
        assert model.model_name == "deepseek-reasoner"

    def test_error_message_lists_available_providers(self, fake_toml):
        with pytest.raises(KeyError, match="deepseek"):
            get_llm_client("nonexistent")

    def test_google_config_preserves_raw_api_key(self, fake_toml):
        result = load_providers()
        assert result["google"].api_key == "${GOOGLE_API_KEY}"

    def test_tools_section_ignored(self, fake_toml):
        """The [tools.*] section must not appear in the providers mapping."""
        result = load_providers()
        assert "mcp" not in result

    def test_empty_providers_returns_empty_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
        with patch("builtins.open", mock_open(read_data=b"")), patch("tomllib.load", return_value={}):
            result = load_providers()
        assert result == {}
