"""
LLM provider factory.

Parses ``infra.toml`` at the repo root and builds a LangChain chat-model
instance for every entry under ``[providers.*]``.

Supported types
---------------
rest          – OpenAI-compatible REST API (``langchain-openai``)
google_genai  – Google Generative AI      (``langchain-google-genai``)
"""

import os
import re
import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

# infra.toml lives at the repo root (three levels up from this file:
#   src/agent/providers.py  →  src/agent/  →  src/  →  <repo root>)
_INFRA_TOML = Path(__file__).parent.parent.parent / "infra.toml"


def _expand_env(value: str) -> str:
    """Replace ``${VAR}`` placeholders with the matching environment variable."""
    return re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1), ""), value)


# ---------------------------------------------------------------------------
# Provider config models
# ---------------------------------------------------------------------------


class RestProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["rest"] = "rest"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GoogleGenAIProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["google_genai"]
    api_key: str = ""
    model: str = "gemini-2.0-flash"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


ProviderConfig = Annotated[
    Union[RestProviderConfig, GoogleGenAIProviderConfig],
    Field(discriminator="type"),
]

_provider_adapter: TypeAdapter[RestProviderConfig | GoogleGenAIProviderConfig] = TypeAdapter(ProviderConfig)


def _parse_provider_config(raw: dict[str, Any]) -> RestProviderConfig | GoogleGenAIProviderConfig:
    """Validate and coerce a raw config dict into a typed provider config.

    Defaults ``type`` to ``"rest"`` when not present.
    Raises ``pydantic.ValidationError`` on invalid data.
    """
    return _provider_adapter.validate_python({"type": "rest", **raw})


# ---------------------------------------------------------------------------
# Provider builder
# ---------------------------------------------------------------------------


def _get_llm_client_from_cfg(cfg: RestProviderConfig | GoogleGenAIProviderConfig, **kwargs: Any) -> BaseChatModel:
    if isinstance(cfg, RestProviderConfig):
        api_key = _expand_env(cfg.api_key) or "dummy"
        base_url = _expand_env(cfg.base_url) or None
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=kwargs["model"] if "model" in kwargs else cfg.model,
            temperature=kwargs["temperature"] if "temperature" in kwargs else cfg.temperature,
        )

    # GoogleGenAIProviderConfig
    api_key = _expand_env(cfg.api_key) or None
    kwargs_model: dict[str, Any] = {
        "model": kwargs["model"] if "model" in kwargs else cfg.model,
        "temperature": kwargs["temperature"] if "temperature" in kwargs else cfg.temperature,
    }
    if api_key:
        kwargs_model["google_api_key"] = api_key
    return ChatGoogleGenerativeAI(**kwargs_model)


def load_providers() -> dict[str, RestProviderConfig | GoogleGenAIProviderConfig]:
    """Parse ``infra.toml`` and return a ``{name: provider_config}`` mapping."""
    with open(_INFRA_TOML, "rb") as f:
        config = tomllib.load(f)

    return {name: _parse_provider_config(raw) for name, raw in config.get("providers", {}).items()}


def get_llm_client(name: str, **kwargs: Any) -> BaseChatModel:
    """Return the chat model for the provider with the given name.

    Raises ``KeyError`` if no provider with that name exists in ``infra.toml``.
    """
    providers = load_providers()
    if name not in providers:
        raise KeyError(f"Provider '{name}' not found. Available: {sorted(providers)}")
    return _get_llm_client_from_cfg(providers[name], **kwargs)
