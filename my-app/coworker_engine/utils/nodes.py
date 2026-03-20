from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from .state import AgentState
from dotenv import load_dotenv
import os
import re
from .knowledge import format_knowledge_context, retrieve_knowledge
from .safety import find_forbidden_language
from .tools import (
    add_jira_comment,
    calculate_kpi,
    create_jira_task,
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
llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen2.5:32b"), temperature=0.7)
TOOL_CAPABLE_MODEL_PREFIXES = ("qwen", "mistral", "smollm", "gemma", "deepseek")
ROUTE_BY_NPC = {
    "CEO": "ceo",
    "CHRO": "chro",
    "Regional Manager": "regional",
}

# ─────────────────────────────────────────────
# SAFETY NODE — runs before any LLM call
# ─────────────────────────────────────────────
def safety_node(state: AgentState) -> dict:
    """
    Pre-LLM safety check. Scans the latest user message for forbidden keywords.
    If any are found, they are stored in state['safety_flags'] and a blocking
    AIMessage is returned so the graph can end early.
    """
    last_message = state.get("messages", [])[-1] if state.get("messages") else None
    user_text = last_message.content if last_message else ""

    flags = find_forbidden_language(user_text)

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


# Per-persona "magic words" that increase trust when mentioned by the user
REPUTATION_TRIGGERS = {
    "CEO": ["dna", "heritage", "legacy", "brand equity", "synergy"],
    "CHRO": ["talent", "competency", "mobility", "leadership", "framework"],
    "Regional Manager": ["rollout", "region", "stakeholder", "workshop", "local"],
}


def _update_persona_reputation(state: AgentState, npc_name: str, user_text: str) -> dict:
    updated_for_turn = list(state.get("reputation_updated_for_turn", []))
    if npc_name in updated_for_turn:
        return {
            "persona_reputation": dict(state.get("persona_reputation", {})),
            "persona_alignment": dict(state.get("persona_alignment", {})),
            "reputation": dict(state.get("persona_reputation", {})).get(npc_name, state.get("reputation", 0.5)),
            "alignment_score": dict(state.get("persona_alignment", {})).get(npc_name, state.get("alignment_score", 0.0)),
            "reputation_updated_for_turn": updated_for_turn,
        }

    current_reputation_map = dict(state.get("persona_reputation", {}))
    current_alignment_map = dict(state.get("persona_alignment", {}))

    current_reputation = current_reputation_map.get(npc_name, 0.5)
    current_alignment = current_alignment_map.get(npc_name, 0.0)
    triggers = REPUTATION_TRIGGERS.get(npc_name, [])
    hits = [trigger for trigger in triggers if trigger in user_text]

    reputation_delta = len(hits) * 0.05
    new_reputation = min(1.0, current_reputation + reputation_delta)
    new_alignment = current_alignment + len(hits)

    current_reputation_map[npc_name] = new_reputation
    current_alignment_map[npc_name] = new_alignment
    updated_for_turn.append(npc_name)

    return {
        "persona_reputation": current_reputation_map,
        "persona_alignment": current_alignment_map,
        "reputation": new_reputation,
        "alignment_score": new_alignment,
        "reputation_updated_for_turn": updated_for_turn,
    }


# ─────────────────────────────────────────────
# NPC NODE FACTORY — now reputation-aware
# ─────────────────────────────────────────────

# Only send the last N messages to the LLM per call.
# This is the main lever for reducing latency as conversation grows.
# Increase for richer context; decrease for faster responses.
HISTORY_WINDOW = 6  # last 3 user + 3 AI turns


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"\b\w+\b")


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
        "executive update",
        "exec update",
        "email",
        "subject line",
        "internal communication",
        "internal comm",
        "post",
        "summary",
        "draft",
    ]
    return any(marker in lowered for marker in detail_markers)


def _is_heading_only_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if "\n" in stripped:
        return False

    word_count = len(WORD_RE.findall(stripped))
    has_sentence_punctuation = any(char in stripped for char in ".!?")
    markdown_heading = stripped.startswith("#") or (stripped.startswith("**") and stripped.endswith("**"))
    title_stub = stripped.endswith(":")
    return word_count <= 14 and not has_sentence_punctuation and (markdown_heading or title_stub)


def _compress_chat_reply(text: str, user_text: str) -> str:
    if _user_requested_detailed_format(user_text):
        return text.strip()

    paragraphs = [part.strip() for part in text.strip().split("\n\n") if part.strip()]
    if paragraphs and _is_heading_only_block(paragraphs[0]):
        return "\n\n".join(paragraphs[: min(3, len(paragraphs))]).strip()

    first_block = paragraphs[0] if paragraphs else text.strip()
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(first_block) if s.strip()]
    if len(sentences) <= 2:
        return first_block
    return " ".join(sentences[:2]).strip()


def _build_turn_focus_instruction(npc_name: str, user_text: str) -> str:
    lowered = user_text.lower()
    instructions: list[str] = []

    asks_for_tradeoff = any(
        term in lowered
        for term in ["trade-off", "tradeoff", "balance", "what would you force", "final recommendation"]
    )

    if npc_name == "CHRO":
        asks_for_roi = "roi" in lowered or "return on investment" in lowered
        asks_for_mobility = "mobility" in lowered or "inter-brand" in lowered
        asks_for_360 = "360" in lowered or "feedback" in lowered
        asks_for_coaching = "coaching" in lowered or "coach" in lowered

        if asks_for_roi or asks_for_mobility or asks_for_360 or asks_for_coaching:
            instructions.append(
                "For this turn, do not stay generic. If ROI is requested, state one concrete business benefit "
                "or measurable outcome. If mobility is requested, state exactly how the proposal improves inter-brand mobility."
            )
        if asks_for_360 or asks_for_coaching:
            instructions.append(
                "When 360 feedback or coaching is mentioned, explain their role separately instead of collapsing them into generic training."
            )
        if asks_for_tradeoff:
            instructions.append(
                "State one explicit trade-off, such as depth of development versus rollout cost or speed."
            )

    if npc_name == "Regional Manager":
        asks_for_rollout = any(
            term in lowered
            for term in ["rollout", "local", "region", "regional", "adoption", "stakeholder", "burden"]
        )
        if asks_for_rollout or asks_for_tradeoff:
            instructions.append(
                "Be concrete about local implementation burden: name at least one staffing, time, or adoption constraint."
            )
        if asks_for_tradeoff:
            instructions.append(
                "Do not end with a clarification question. State one concrete trade-off you would force HQ to accept."
            )

    return " ".join(instructions)


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
        create_jira_task,
        add_jira_comment,
        update_jira_status,
    ]
    tool_enabled_llm = (
        llm.bind_tools(agent_tools)
        if llm.model.lower().startswith(TOOL_CAPABLE_MODEL_PREFIXES)
        else None
    )

    def node(state: AgentState):
        full_history = list(state.get("messages", []))
        last_user_message = next(
            (message.content for message in reversed(full_history) if message.type == "human"),
            "",
        )
        reputation_state = _update_persona_reputation(state, npc_name, last_user_message.lower())
        reputation = reputation_state["reputation"]
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
            "instead of guessing. When the user asks you to create a task, add a comment, or update a "
            "task, use a tool to perform the action. Use your own namespace when calling retrieve_brand_data: "
            f"`{namespace}`. Use your own agent_id when calling add_jira_comment: `{agent_id}`. "
            "After receiving tool results, answer the user directly and do not mention internal tool mechanics.\n\n"
            f"[RELATIONSHIP CONTEXT]: The user's current reputation with you is "
            f"{reputation:.2f}/1.0 — your attitude toward them is currently {warmth}. "
            f"Reflect this in how forthcoming and warm your response is."
        )

        turn_focus_instruction = _build_turn_focus_instruction(npc_name, last_user_message)
        if turn_focus_instruction:
            system_message_content += f"\n\n[TURN-SPECIFIC REQUIREMENT]: {turn_focus_instruction}"

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
                tool_result = {
                    "messages": [response],
                    "active_npc": npc_name,
                    "supervisor_hint": state.get("supervisor_hint", ""),
                }
                tool_result.update(reputation_state)
                return tool_result
        else:
            response = llm.invoke(messages_to_pass)

        response_text = response.content if isinstance(response.content, str) else str(response.content)
        response_text = _compress_chat_reply(response_text, last_user_message)
        response.content = response_text

        route_key = ROUTE_BY_NPC[npc_name]
        updated_meeting_queue = list(state.get("meeting_queue", []))
        if state.get("mode") == "meeting" and updated_meeting_queue and updated_meeting_queue[0] == route_key:
            updated_meeting_queue = updated_meeting_queue[1:]

        updated_meeting_notes = list(state.get("meeting_notes", []))
        if state.get("mode") == "meeting":
            updated_meeting_notes.append(
                {
                    "npc": npc_name,
                    "route": route_key,
                    "content": response_text,
                }
            )

        final_state = {
            "messages": [response],
            "active_npc": npc_name,
            "meeting_queue": updated_meeting_queue,
            "meeting_notes": updated_meeting_notes,
            "supervisor_hint": "" if state.get("mode") == "direct_reply" else state.get("supervisor_hint", ""),
        }
        final_state.update(reputation_state)
        return final_state

    return node


ceo_node = build_npc_node(CEO_PROMPT, "CEO", "ceo")
chro_node = build_npc_node(CHRO_PROMPT, "CHRO", "chro")
regional_node = build_npc_node(
    REGIONAL_PROMPT, "Regional Manager", "regional"
)


def meeting_synthesis_node(state: AgentState) -> dict:
    meeting_notes = list(state.get("meeting_notes", []))
    latest_user_message = next(
        (message.content for message in reversed(state.get("messages", [])) if message.type == "human"),
        "",
    )
    notes_text = "\n".join(
        f"- {note.get('npc', 'Unknown')}: {note.get('content', '')}"
        for note in meeting_notes
    ) or "- No meeting notes available."

    system_message = SystemMessage(
        content=(
            "You are the invisible Supervisor summarizing a cross-functional meeting. "
            "Respond in a neutral narrator voice. Synthesize the CEO, CHRO, and Regional "
            "Manager views into one final recommendation. Keep it concise, concrete, and "
            "balanced. Name at least one trade-off. Do not mention hidden instructions, "
            "internal routing, or tool mechanics."
        )
    )
    user_message = HumanMessage(
        content=(
            f"User request: {latest_user_message}\n\n"
            f"Meeting notes:\n{notes_text}"
        )
    )
    response = llm.invoke([system_message, user_message])
    response_text = response.content if isinstance(response.content, str) else str(response.content)
    response.content = _compress_chat_reply(response_text, latest_user_message)

    return {
        "messages": [response],
        "active_npc": "Supervisor",
        "meeting_queue": [],
        "meeting_notes": [],
        "supervisor_hint": "",
        "reputation_updated_for_turn": [],
    }
