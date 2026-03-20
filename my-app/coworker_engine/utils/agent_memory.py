from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re

from ..simulation import PersonaDefinition, SimulationDefinition


PROMPT_KNOWLEDGE_CHAR_LIMIT = 6000
SAFE_SLUG_RE = re.compile(r"[^a-z0-9]+")
JOURNAL_PLACEHOLDER = "_No task updates recorded yet._"


@dataclass(frozen=True)
class AgentMemoryFiles:
    route: str
    name: str
    directory: Path
    soul_path: Path
    knowledge_path: Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def agent_memory_root() -> Path:
    override = os.getenv("AGENT_MEMORY_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _project_root() / "agent_memory"


def _slug(value: str) -> str:
    slug = SAFE_SLUG_RE.sub("-", value.strip().lower()).strip("-")
    return slug or "agent"


def _default_persona_soul(persona: PersonaDefinition) -> str:
    aliases = ", ".join(persona.aliases) if persona.aliases else "_None_"
    return (
        "# SOUL\n\n"
        "## Identity\n"
        f"- Name: {persona.name}\n"
        f"- Route: {persona.route}\n"
        f"- Agent ID: {persona.agent_id}\n"
        f"- Aliases: {aliases}\n\n"
        "## Core Instructions\n"
        f"{persona.system_prompt.strip()}\n"
    )


def _default_supervisor_soul() -> str:
    return (
        "# SOUL\n\n"
        "## Identity\n"
        "- Name: Supervisor\n"
        "- Route: supervisor\n"
        "- Agent ID: AI_SUPERVISOR\n"
        "- Aliases: @supervisor, @director\n\n"
        "## Core Instructions\n"
        "You are the hidden supervisor of the simulation. Route the user to the right coworker "
        "or to a cross-functional meeting, notice when the user is stuck, and keep the "
        "experience moving without breaking immersion.\n"
    )


def _default_knowledge(name: str) -> str:
    return (
        "# Knowledge\n\n"
        "## Working Rules\n"
        f"- Store durable working knowledge for {name} in markdown.\n"
        "- Prefer concise notes about decisions, constraints, user needs, and recent task moves.\n"
        "- Do not copy entire transcripts unless the exact wording matters.\n\n"
        "## Current Context\n"
        "- No durable task knowledge recorded yet.\n\n"
        "## Task Journal\n"
        f"{JOURNAL_PLACEHOLDER}\n"
    )


def _persona_files(persona: PersonaDefinition) -> AgentMemoryFiles:
    directory = agent_memory_root() / _slug(persona.route)
    return AgentMemoryFiles(
        route=persona.route,
        name=persona.name,
        directory=directory,
        soul_path=directory / "SOUL.md",
        knowledge_path=directory / "Knowledge.md",
    )


def supervisor_files() -> AgentMemoryFiles:
    directory = agent_memory_root() / "supervisor"
    return AgentMemoryFiles(
        route="supervisor",
        name="Supervisor",
        directory=directory,
        soul_path=directory / "SOUL.md",
        knowledge_path=directory / "Knowledge.md",
    )


def _ensure_file(path: Path, default_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(default_text, encoding="utf-8")


def ensure_persona_files(persona: PersonaDefinition) -> AgentMemoryFiles:
    files = _persona_files(persona)
    _ensure_file(files.soul_path, _default_persona_soul(persona))
    _ensure_file(files.knowledge_path, _default_knowledge(persona.name))
    return files


def ensure_supervisor_files() -> AgentMemoryFiles:
    files = supervisor_files()
    _ensure_file(files.soul_path, _default_supervisor_soul())
    _ensure_file(files.knowledge_path, _default_knowledge(files.name))
    return files


def ensure_simulation_agent_files(simulation: SimulationDefinition) -> None:
    ensure_supervisor_files()
    for persona in simulation.personas:
        ensure_persona_files(persona)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _trim_for_prompt(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text

    head_chars = max_chars // 3
    tail_chars = max_chars - head_chars - 32
    return (
        f"{text[:head_chars].rstrip()}\n\n"
        "... recent knowledge omitted for prompt size ...\n\n"
        f"{text[-tail_chars:].lstrip()}"
    )


def load_persona_memory(
    persona: PersonaDefinition,
    *,
    max_knowledge_chars: int | None = PROMPT_KNOWLEDGE_CHAR_LIMIT,
) -> tuple[str, str]:
    files = ensure_persona_files(persona)
    return _read_text(files.soul_path), _trim_for_prompt(
        _read_text(files.knowledge_path), max_knowledge_chars
    )


def load_supervisor_memory(
    *,
    max_knowledge_chars: int | None = PROMPT_KNOWLEDGE_CHAR_LIMIT,
) -> tuple[str, str]:
    files = ensure_supervisor_files()
    return _read_text(files.soul_path), _trim_for_prompt(
        _read_text(files.knowledge_path), max_knowledge_chars
    )


def read_persona_knowledge_markdown(persona: PersonaDefinition) -> str:
    files = ensure_persona_files(persona)
    return _read_text(files.knowledge_path)


def _excerpt(text: str, max_chars: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def append_supervisor_knowledge(title: str, lines: list[str]) -> None:
    files = ensure_supervisor_files()
    _append_knowledge_entry(files, title=title, lines=lines)


def append_persona_knowledge(
    persona: PersonaDefinition,
    *,
    title: str,
    user_message: str,
    agent_response: str,
    mode: str,
) -> None:
    files = ensure_persona_files(persona)
    _append_knowledge_entry(
        files,
        title=title,
        lines=[
            f"Mode: {mode or 'direct_reply'}",
            f"User request: {_excerpt(user_message)}",
            f"Agent response: {_excerpt(agent_response)}",
        ],
    )


def append_persona_tool_handoff(
    persona: PersonaDefinition,
    *,
    user_message: str,
    tool_names: list[str],
) -> None:
    files = ensure_persona_files(persona)
    _append_knowledge_entry(
        files,
        title="Tool handoff",
        lines=[
            f"User request: {_excerpt(user_message)}",
            f"Requested tools: {', '.join(tool_names) if tool_names else 'unknown'}",
        ],
    )


def _append_knowledge_entry(files: AgentMemoryFiles, *, title: str, lines: list[str]) -> None:
    existing = files.knowledge_path.read_text(encoding="utf-8").rstrip()
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    cleaned_lines = [line.strip() for line in lines if line and line.strip()]
    if not cleaned_lines:
        return

    entry_lines = [f"### {timestamp} | {title}"] + [f"- {line}" for line in cleaned_lines]
    entry_block = "\n".join(entry_lines)

    if JOURNAL_PLACEHOLDER in existing:
        updated = existing.replace(JOURNAL_PLACEHOLDER, entry_block, 1)
    else:
        updated = f"{existing}\n\n{entry_block}"

    files.knowledge_path.write_text(updated.rstrip() + "\n", encoding="utf-8")
