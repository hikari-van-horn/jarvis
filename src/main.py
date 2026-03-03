"""
Jarvis application entry point.

Behaviour
---------
- The **MCP server** is **always** started (background thread, localhost:8001).
- Extensions (discord, telegram, …) are **optional** — pass ``exts=<name,...>``
  on the command line to enable them.
- If no extensions are requested the process stays alive so the MCP server
  keeps running until the user presses Ctrl+C.

Examples
--------
Run only the MCP server (no bot)::

    python -m src.main

Run the MCP server *and* the Discord bot::

    python -m src.main exts=discord

Run the MCP server *and* multiple bots::

    python -m src.main exts=discord,telegram
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import sys
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("main")

# Ensure the project root is on PYTHONPATH so ``src.*`` imports work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MCP_HOST =  os.getenv('MCP_HOST', "127.0.0.1")
MCP_PORT = int(os.getenv('MCP_PORT', 8001))
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}"


# ---------------------------------------------------------------------------
# MCP server lifecycle
# ---------------------------------------------------------------------------


def _run_mcp_server_thread() -> None:
    """Target for the MCP-server background thread (blocking call)."""
    try:
        from src.tools.server import run as run_mcp

        run_mcp(host=MCP_HOST, port=MCP_PORT)
    except Exception as exc:
        logger.error("MCP server crashed: %s", exc)


async def start_mcp_server() -> None:
    """Launch the MCP server in a daemon thread and wait until it's reachable."""
    thread = threading.Thread(
        target=_run_mcp_server_thread,
        daemon=True,
        name="mcp-server",
    )
    thread.start()
    logger.info("MCP server thread started — waiting for it to be reachable …")

    # Poll the SSE endpoint until it responds (max ~10 s).
    import httpx

    for attempt in range(20):
        await asyncio.sleep(0.5)
        try:
            async with httpx.AsyncClient() as client:
                # Use stream() so we get the response headers without reading
                # the infinite SSE body — close immediately once headers arrive.
                async with client.stream("GET", f"{MCP_URL}/sse", timeout=2.0) as r:
                    if r.status_code == 200:
                        logger.info(
                            "MCP server is reachable at %s (attempt %d)",
                            MCP_URL,
                            attempt + 1,
                        )
                        return
        except Exception:
            pass

    logger.warning(
        "MCP server did not respond within 10 s — continuing anyway. "
        "Tool calls may fail until it is ready."
    )


# ---------------------------------------------------------------------------
# Extension lifecycle
# ---------------------------------------------------------------------------


async def run_extensions(ext_names: list[str]) -> None:
    """Import and start each named extension concurrently."""
    tasks: list[asyncio.Task] = []

    for ext_name in ext_names:
        module_name = f"src.exts.{ext_name}_ext"
        try:
            ext_module = importlib.import_module(module_name)
            if hasattr(ext_module, "start") and asyncio.iscoroutinefunction(
                ext_module.start
            ):
                logger.info("Starting extension: %s", ext_name)
                tasks.append(asyncio.create_task(ext_module.start()))
            else:
                logger.error(
                    "Extension '%s' has no async 'start' function — skipping.",
                    ext_name,
                )
        except ImportError as exc:
            logger.error("Could not import extension '%s': %s", ext_name, exc)
        except Exception as exc:
            logger.error("Error initialising extension '%s': %s", ext_name, exc)

    if tasks:
        logger.info("Running %d extension(s) concurrently …", len(tasks))
        await asyncio.gather(*tasks)
    else:
        logger.info(
            "No extensions running. MCP server is active on %s. "
            "Press Ctrl+C to quit.",
            MCP_URL,
        )
        # Keep the event loop alive so the daemon MCP thread stays up.
        await asyncio.Event().wait()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _init_db_schema() -> None:
    """Apply SurrealDB schema (idempotent — safe to run on every startup)."""
    try:
        from src.agent.memory.store import MemoryStore
        from src.agent.memory.conversation_store import ConversationStore

        store = MemoryStore()
        await store.connect()
        await store.init_schema()
        await store.close()

        conv = ConversationStore()
        await conv.connect()
        await conv.init_schema()
        await conv.close()

        logger.info("SurrealDB schema initialised successfully.")
    except Exception as exc:
        logger.error("SurrealDB schema init failed: %s", exc)


async def _main(ext_names: list[str]) -> None:
    # 1. Apply DB schema (OVERWRITE — safe on every startup).
    await _init_db_schema()
    # 2. Start the MCP server.
    await start_mcp_server()
    # 3. Optionally start extensions.
    await run_extensions(ext_names)


def main() -> None:
    # Parse ``exts=discord,telegram`` from argv; default = no extensions.
    ext_names: list[str] = []
    for arg in sys.argv[1:]:
        if arg.startswith("exts="):
            ext_names = [
                e.strip() for e in arg[len("exts="):].split(",") if e.strip()
            ]

    try:
        asyncio.run(_main(ext_names))
    except KeyboardInterrupt:
        logger.info("Jarvis shut down by user.")


if __name__ == "__main__":
    main()
