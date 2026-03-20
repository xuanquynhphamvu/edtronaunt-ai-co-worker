from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from .state import AgentState
from dotenv import load_dotenv
import re
from .knowledge import format_knowledge_context, retrieve_knowledge
from .tools import (
    add_jira_comment,
    calculate_kpi,
    list_jira_tasks,
    retrieve_brand_data,
    search_jira_tasks,
    update_jira_status,
)

load_dotenv()

from ..personas.ceo import CEO_PROMPT
from ..personas.chro import CHRO_PROMPT
from ..personas.regional import REGIONAL_PROMPT

# Initialize Ollama model
llm = ChatOllama(model="llama3", temperature=0.7)
TOOL_CAPABLE_MODEL_PREFIXES = ("qwen", "mistral", "smollm", "gemma", "deepseek")

# ─────────────────────────────────────────────
# SAFETY NODE — runs before any LLM call
# ─────────────────────────────────────────────
# Match only whole forbidden words so innocuous terms like "better"
# do not trip the safety filter via substring collisions.
GLOBAL_FORBIDDEN_PATTERNS = {
    "bet": re.compile(r"\bbet(?:s|ting)?\b", re.IGNORECASE),
    "gamble": re.compile(r"\bgambl(?:e|es|ed|ing)\b", re.IGNORECASE),
    "emoji": re.compile(r"\bemojis?\b", re.IGNORECASE),
    "wager": re.compile(r"\bwager(?:s|ed|ing)?\b", re.IGNORECASE),
    "stake": re.compile(r"\bstakes?\b", re.IGNORECASE),
}


def safety_node(state: AgentState) -> dict:
    """
    Pre-LLM safety check. Scans the latest user message for forbidden keywords.
    If any are found, they are stored in state['safety_flags'] and a blocking
    AIMessage is returned so the graph can end early.
    """
    last_message = state.get("messages", [])[-1] if state.get("messages") else None
    user_text = last_message.content if last_message else ""

    flags = [
        f"Forbidden keyword detected: '{kw}'"
        for kw, pattern in GLOBAL_FORBIDDEN_PATTERNS.items()
        if pattern.search(user_text)
    ]

    if flags:
        block_response = AIMessage(
            content=(
                "I'm afraid that type of language is not appropriate in this context. "
                "Let us refocus on the matter at hand."
            )
        )
        return {
            "safety_flags": flags,
            "messages": [block_response],
            "next_route": "end",  # Signal to supervisor to halt routing
        }

    return {"safety_flags": []}


# ─────────────────────────────────────────────
# REPUTATION NODE — runs before any LLM call
# ─────────────────────────────────────────────
# Per-persona "magic words" that increase trust when mentioned by the user
REPUTATION_TRIGGERS = {
    "CEO": ["dna", "heritage", "legacy", "brand equity", "synergy"],
    "CHRO": ["talent", "competency", "mobility", "leadership", "framework"],
    "Regional Manager": ["rollout", "region", "stakeholder", "workshop", "local"],
}


def reputation_node(state: AgentState) -> dict:
    """
    Reads the latest user message and increases the reputation score if the user
    mentions keywords aligned with the active NPC's values. Caps at 1.0 / floors at 0.0.
    Also increments the turn count and recalculates the alignment score.
    """
    last_message = state.get("messages", [])[-1] if state.get("messages") else None
    user_text = last_message.content.lower() if last_message else ""

    active_npc = state.get("active_npc", "")
    current_reputation = state.get("reputation", 0.5)
    current_alignment = state.get("alignment_score", 0.0)

    triggers = REPUTATION_TRIGGERS.get(active_npc, [])
    hits = [t for t in triggers if t in user_text]

    # +0.05 per trigger keyword found, capped at 1.0
    reputation_delta = len(hits) * 0.05
    new_reputation = min(1.0, current_reputation + reputation_delta)

    # Alignment score: cumulative count of keyword hits
    new_alignment = current_alignment + len(hits)

    return {
        "reputation": new_reputation,
        "alignment_score": new_alignment,
        "turn_count": state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────
# NPC NODE FACTORY — now reputation-aware
# ─────────────────────────────────────────────

# Only send the last N messages to the LLM per call.
# This is the main lever for reducing latency as conversation grows.
# Increase for richer context; decrease for faster responses.
HISTORY_WINDOW = 6  # last 3 user + 3 AI turns


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _user_requested_detailed_format(user_text: str) -> bool:
    lowered = user_text.lower()
    detail_markers = [
        "plan",
        "steps",
        "bullet",
        "bullets",
        "list",
        "framework",
        "detailed",
        "detail",
        "roadmap",
        "break down",
        "breakdown",
    ]
    return any(marker in lowered for marker in detail_markers)


def _compress_chat_reply(text: str, user_text: str) -> str:
    if _user_requested_detailed_format(user_text):
        return text.strip()

    paragraphs = [part.strip() for part in text.strip().split("\n\n") if part.strip()]
    first_block = paragraphs[0] if paragraphs else text.strip()
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(first_block) if s.strip()]
    if len(sentences) <= 2:
        return first_block
    return " ".join(sentences[:2]).strip()


def build_npc_node(prompt: str, npc_name: str, namespace: str):
    """Factory function to build node logic for a specific NPC."""

    agent_id = {
        "CEO": "AI_CEO",
        "CHRO": "AI_CHRO",
        "Regional Manager": "AI_REGIONAL",
    }.get(npc_name, "AI_AGENT")

    agent_tools = [
        calculate_kpi,
        retrieve_brand_data,
        list_jira_tasks,
        search_jira_tasks,
        add_jira_comment,
        update_jira_status,
    ]
    tool_enabled_llm = (
        llm.bind_tools(agent_tools)
        if llm.model.lower().startswith(TOOL_CAPABLE_MODEL_PREFIXES)
        else None
    )

    def node(state: AgentState):
        reputation = state.get("reputation", 0.5)
        full_history = list(state.get("messages", []))
        last_user_message = next(
            (message.content for message in reversed(full_history) if message.type == "human"),
            "",
        )
        knowledge_chunks = retrieve_knowledge(last_user_message, namespaces=[namespace], top_k=3)

        # Map reputation score to a human-readable warmth level for the LLM
        if reputation >= 0.8:
            warmth = "highly trusting and open"
        elif reputation >= 0.5:
            warmth = "neutral and professional"
        else:
            warmth = "skeptical and guarded"

        system_message_content = (
            f"{prompt}\n"
            f"Answer carefully as the selected role: {npc_name}.\n\n"
            "Keep the answer natural and human-sized. Default to one short paragraph with 2 to 4 "
            "sentences. Do not write a consultant-style memo. Do not give a multi-step framework "
            "unless the user explicitly asks for one. If the user asks a broad question, give the "
            "single most useful answer first.\n\n"
            "This Gucci scenario is not only about describing a leadership model. When the user asks "
            "for a recommendation, framework, or design, ground your answer in the real case tension: "
            "protect brand DNA, improve talent development and inter-brand mobility, use 360 feedback "
            "plus coaching credibly, and acknowledge regional rollout constraints. Do not pretend every "
            "goal can be maximized at once; state at least one concrete trade-off when relevant.\n\n"
            "When the user asks about live tasks, comments, backlog, or status, use the Jira tools "
            "instead of guessing. When the user asks you to add a comment or update a task, use a tool "
            "to perform the action. Use your own namespace when calling retrieve_brand_data: "
            f"`{namespace}`. Use your own agent_id when calling add_jira_comment: `{agent_id}`. "
            "After receiving tool results, answer the user directly and do not mention internal tool mechanics.\n\n"
            f"[RELATIONSHIP CONTEXT]: The user's current reputation with you is "
            f"{reputation:.2f}/1.0 — your attitude toward them is currently {warmth}. "
            f"Reflect this in how forthcoming and warm your response is."
        )

        if knowledge_chunks:
            system_message_content += (
                "\n\n[RETRIEVED SIMULATION CONTEXT]:\n"
                f"{format_knowledge_context(knowledge_chunks)}"
                "\nUse this retrieved context when it is relevant. If the context is incomplete, "
                "make the smallest reasonable assumption and say what you are assuming. "
                "Do not dump all retrieved context back to the user. Never mention sources, files, "
                "briefs, or retrieved context unless the user explicitly asks how you know."
            )

        # Check if the supervisor injected a hidden hint for this turn
        hint = state.get("supervisor_hint", "")
        if hint:
            system_message_content += (
                f"\n\n[SECRET INSTRUCTION FROM SUPERVISOR]: {hint}"
            )

        # LLM system message context
        system_message = SystemMessage(content=system_message_content)

        # Trim history to the last HISTORY_WINDOW messages to keep latency low.
        # The system prompt is always prepended fresh, so persona context is never lost.
        windowed_history = full_history[-HISTORY_WINDOW:]
        messages_to_pass = [system_message] + windowed_history

        if tool_enabled_llm is not None:
            response = tool_enabled_llm.invoke(messages_to_pass)
            if getattr(response, "tool_calls", None):
                return {
                    "messages": [response],
                    "active_npc": npc_name,
                    "supervisor_hint": "",
                }
        else:
            response = llm.invoke(messages_to_pass)

        response_text = response.content if isinstance(response.content, str) else str(response.content)
        response_text = _compress_chat_reply(response_text, last_user_message)
        response.content = response_text

        return {
            "messages": [response],
            "active_npc": npc_name,
            "supervisor_hint": "",  # Clear the hint after it's used
        }

    return node


ceo_node = build_npc_node(CEO_PROMPT, "CEO", "ceo")
chro_node = build_npc_node(CHRO_PROMPT, "CHRO", "chro")
regional_node = build_npc_node(
    REGIONAL_PROMPT, "Regional Manager", "regional"
)
