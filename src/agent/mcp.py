
import logging
import os
from contextlib import AsyncExitStack

logger = logging.getLogger("mcp")

MCP_SSE_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8001/sse")


class MCPClientManager:
    """Maintains a persistent SSE connection to the local MCP server.

    Tools are loaded once and reused for the lifetime of the agent.
    Call ``get_tools()`` to lazily connect and retrieve the tool list.
    Call ``close()`` to cleanly shut down the connection.
    """

    def __init__(self, url: str = MCP_SSE_URL) -> None:
        self._url = url
        self._tools: list | None = None
        self._exit_stack: AsyncExitStack | None = None

    async def get_tools(self) -> list:
        """Return cached tools, connecting to the MCP server on first call."""
        if self._tools is not None:
            return self._tools
        try:
            from langchain_mcp_adapters.tools import load_mcp_tools
            from mcp import ClientSession
            from mcp.client.sse import sse_client

            stack = AsyncExitStack()
            read, write = await stack.enter_async_context(sse_client(self._url))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._tools = await load_mcp_tools(session)
            self._exit_stack = stack
            logger.info(
                "MCPClientManager: loaded %d tools from %s",
                len(self._tools),
                self._url,
            )
        except Exception as exc:
            logger.warning(
                "MCPClientManager: could not connect to MCP server (%s) — "
                "tool-calling will be disabled for this session.",
                exc,
            )
            self._tools = []
        return self._tools

    async def close(self) -> None:
        """Close the MCP SSE connection."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._tools = None

