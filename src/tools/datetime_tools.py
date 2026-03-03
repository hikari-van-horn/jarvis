"""Date/time tools registered on the shared MCP server instance."""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.tools import mcp

logger = logging.getLogger("tools.datetime")


@mcp.tool()
def get_current_datetime(timezone: str = "UTC") -> str:
    """Get the current date and time in the specified IANA timezone.

    Args:
        timezone: IANA timezone name, e.g. 'America/New_York', 'Asia/Tokyo', 'UTC'.
    """
    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except ZoneInfoNotFoundError:
        logger.warning("Unknown timezone '%s', falling back to UTC", timezone)
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception as exc:
        logger.error("get_current_datetime error: %s", exc)
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
