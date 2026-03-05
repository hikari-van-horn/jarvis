"""Agent skills discovery and loading tools registered on the shared MCP server instance.

Two directories are scanned:
  - Built-in skills : <repo>/src/agent/skills/
  - User skills     : <repo>/data/skills/

Each skill lives in its own sub-directory and must contain a ``SKILL.md`` file
whose YAML front-matter declares at least ``name`` and ``description``.

Example SKILL.md header::

    ---
    name: brainstorming
    description: "Explores user intent and requirements before implementation."
    ---
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from src.tools import mcp

logger = logging.getLogger("tools.skills")

# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------

# Repo root = two levels above this file  (src/tools/skills.py → repo root)
_REPO_ROOT = Path(__file__).resolve().parents[2]

_BUILTIN_SKILLS_DIR = _REPO_ROOT / "src" / "agent" / "skills"
_USER_SKILLS_DIR = _REPO_ROOT / "data" / "skills"

_SKILL_FILENAME = "SKILL.md"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> dict[str, str]:
    """Extract YAML-style front-matter (``--- … ---``) as a plain dict.

    Only handles simple ``key: value`` lines — no nested YAML needed for
    skill metadata.
    """
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return {}
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"')
    return meta


def _iter_skills() -> list[dict[str, str]]:
    """Return a list of skill metadata dicts from both skill directories.

    Each dict contains:
        name        – skill identifier (from front-matter or directory name)
        description – short description
        source      – 'builtin' | 'user'
        path        – absolute path to the SKILL.md file
    """
    skills: list[dict[str, str]] = []

    dirs = [
        (_BUILTIN_SKILLS_DIR, "builtin"),
        (_USER_SKILLS_DIR, "user"),
    ]

    for base_dir, source in dirs:
        if not base_dir.exists():
            continue
        for skill_dir in sorted(base_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / _SKILL_FILENAME
            if not skill_file.exists():
                continue
            try:
                content = skill_file.read_text(encoding="utf-8")
                meta = _parse_frontmatter(content)
                skills.append(
                    {
                        "name": meta.get("name", skill_dir.name),
                        "description": meta.get("description", "(no description)"),
                        "source": source,
                        "path": str(skill_file),
                    }
                )
            except Exception as exc:
                logger.warning("Could not read skill %s: %s", skill_file, exc)

    return skills


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


@mcp.tool()
def find_skills(query: Optional[str] = None) -> str:
    """Search for available agent skills in the built-in and user skill directories.

    Scans ``src/agent/skills/`` (built-in) and ``data/skills/`` (user-created)
    for sub-directories containing a ``SKILL.md`` file.  Results are filtered
    by *query* if provided (case-insensitive substring match against skill name
    and description).

    Args:
        query: Optional keyword to filter skills.  If omitted or empty, all
               available skills are returned.

    Returns:
        A formatted list of matching skills with their name, source, and
        description, or a message when no skills are found.
    """
    logger.info("Finding skills with query: '%s'", query)
    all_skills = _iter_skills()

    if query and query.strip():
        q = query.strip().lower()
        all_skills = [
            s for s in all_skills
            if q in s["name"].lower() or q in s["description"].lower()
        ]

    if not all_skills:
        if query:
            return f"No skills found matching '{query}'."
        return "No skills found in either the built-in or user skill directories."

    lines = ["Available agent skills:\n"]
    for s in all_skills:
        lines.append(f"- **{s['name']}** [{s['source']}]")
        lines.append(f"  {s['description']}")
    return "\n".join(lines)


@mcp.tool()
def load_skill(skill_name: str) -> str:
    """Load the full content of an agent skill by name.

    Searches ``src/agent/skills/`` first (built-in), then ``data/skills/``
    (user-created).  Returns the raw ``SKILL.md`` text so the agent can
    follow the skill's instructions.

    Args:
        skill_name: The exact name (directory name or ``name`` field in
                    front-matter) of the skill to load.

    Returns:
        The full markdown content of the skill's ``SKILL.md`` file, or an
        error message if the skill cannot be found.
    """
    if not skill_name or not skill_name.strip():
        return "Error: skill_name must not be empty."
    
    logger.info("Loading skill: '%s'", skill_name)

    target = skill_name.strip().lower()

    # Search both directories; built-in takes precedence.
    dirs = [
        (_BUILTIN_SKILLS_DIR, "builtin"),
        (_USER_SKILLS_DIR, "user"),
    ]

    for base_dir, source in dirs:
        if not base_dir.exists():
            continue
        for skill_dir in base_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / _SKILL_FILENAME
            if not skill_file.exists():
                continue
            try:
                content = skill_file.read_text(encoding="utf-8")
                meta = _parse_frontmatter(content)
                name_in_meta = meta.get("name", skill_dir.name).lower()
                dir_name = skill_dir.name.lower()
                if target in (name_in_meta, dir_name):
                    logger.info("Loaded skill '%s' from %s (%s)", skill_name, skill_file, source)
                    return content
            except Exception as exc:
                logger.warning("Could not read skill %s: %s", skill_file, exc)

    return (
        f"Skill '{skill_name}' not found.  "
        f"Use find_skills() to list all available skills."
    )
