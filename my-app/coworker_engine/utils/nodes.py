from langchain_core.messages import AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from .state import AgentState
from dotenv import load_dotenv

load_dotenv()

from ..personas.ceo import CEO_PROMPT
from ..personas.chro import CHRO_PROMPT
from ..personas.regional import REGIONAL_PROMPT

# Initialize Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# ─────────────────────────────────────────────
# SAFETY NODE — runs before any LLM call
# ─────────────────────────────────────────────
# Keywords that are forbidden across ALL personas
GLOBAL_FORBIDDEN = ["bet", "gamble", "emoji", "wager", "stake"]


def safety_node(state: AgentState) -> dict:
    """
    Pre-LLM safety check. Scans the latest user message for forbidden keywords.
    If any are found, they are stored in state['safety_flags'] and a blocking
    AIMessage is returned so the graph can end early.
    """
    last_message = state.get("messages", [])[-1] if state.get("messages") else None
    user_text = last_message.content.lower() if last_message else ""

    flags = [
        f"Forbidden keyword detected: '{kw}'"
        for kw in GLOBAL_FORBIDDEN
        if kw in user_text
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


def build_npc_node(prompt: str, npc_name: str, namespace: str):
    """Factory function to build node logic for a specific NPC."""

    def node(state: AgentState):
        reputation = state.get("reputation", 0.5)

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
            f"[RELATIONSHIP CONTEXT]: The user's current reputation with you is "
            f"{reputation:.2f}/1.0 — your attitude toward them is currently {warmth}. "
            f"Reflect this in how forthcoming and warm your response is."
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
        full_history = list(state.get("messages", []))
        windowed_history = full_history[-HISTORY_WINDOW:]
        messages_to_pass = [system_message] + windowed_history

        # Invoke Gemini!
        response = llm.invoke(messages_to_pass)

        return {
            "messages": [response],
            "active_npc": npc_name,
            "supervisor_hint": "",  # Clear the hint after it's used
        }

    return node


ceo_node = build_npc_node(CEO_PROMPT, "CEO", "namespace_ceo")
chro_node = build_npc_node(CHRO_PROMPT, "CHRO", "namespace_chro")
regional_node = build_npc_node(
    REGIONAL_PROMPT, "Regional Manager", "namespace_regional"
)
