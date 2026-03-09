# Jarvis

[![CI](https://github.com/hikari-van-horn/jarvis/actions/workflows/ci.yml/badge.svg)](https://github.com/hikari-van-horn/jarvis/actions/workflows/ci.yml)

Jarvis is a conversational AI agent that lives on Discord. It is built on top of [LangGraph](https://github.com/langchain-ai/langgraph) and supports multiple LLM providers (OpenAI, Google Gemini, …). Tool use is powered by the [Model Context Protocol (MCP)](https://modelcontextprotocol.io), and long-term memory is persisted in SurrealDB with semantic search backed by Qdrant.

## Features

- **Discord channel** — chat with Jarvis directly inside any Discord server or DM.
- **Multi-provider LLM** — switch between OpenAI and Google Gemini via config.
- **MCP tool server** — exposes skills (datetime, memory, brainstorming, …) over a local MCP-compatible HTTP server.
- **Persistent memory** — per-user conversation history and knowledge extraction stored in SurrealDB.
- **Semantic search** — vector embeddings in Qdrant power knowledge retrieval.

## Quick Start

```bash
# 1. Install dependencies
poetry install

# 2. Copy and fill in environment variables
cp .env.example .env

# 3. Start backing services (SurrealDB + Qdrant)
docker compose up -d

# 4. Run the agent with the Discord extension
python -m src.main exts=discord
```

## Running Tests

```bash
poetry run pytest tests/ -v
```
