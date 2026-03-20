from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from ..simulation import ACTIVE_SIMULATION, PersonaDefinition
from .agent_memory import (
    append_persona_knowledge,
    append_persona_tool_handoff,
    ensure_simulation_agent_files,
    load_persona_memory,
)
from .knowledge import format_knowledge_context, retrieve_knowledge
from .safety import find_forbidden_language
from .state import AgentState, VisibleResponse
from .tools import (
    add_jira_comment,
    calculate_kpi,
    create_jira_task,
    list_jira_tasks,
    retrieve_simulation_context,
    search_jira_tasks,
    update_jira_status,
)

load_dotenv()

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen2.5:32b"), temperature=0.7)
TOOL_CAPABLE_MODEL_PREFIXES = ("qwen", "mistral", "smollm", "gemma", "deepseek")
PERSONA_BY_ROUTE = {persona.route: persona for persona in ACTIVE_SIMULATION.personas}
ROUTE_BY_NPC = {persona.name: persona.route for persona in ACTIVE_SIMULATION.personas}
REPUTATION_TRIGGERS = {
    persona.name: list(persona.reputation_triggers) for persona in ACTIVE_SIMULATION.personas
}
ensure_simulation_agent_files(ACTIVE_SIMULATION)

HISTORY_WINDOW = 6
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"\b\w+\b")


def safety_node(state: AgentState) -> dict:
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
            "visible_responses": [
                {
                    "speaker": "",
                    "content": block_response.content,
                    "is_final": True,
                }
            ],
            "next_route": "end",
        }

    return {"safety_flags": []}


def _update_persona_reputation(state: AgentState, npc_name: str, user_text: str) -> dict:
    updated_for_turn = list(state.get("reputation_updated_for_turn", []))
    if npc_name in updated_for_turn:
        return {
            "persona_reputation": dict(state.get("persona_reputation", {})),
            "persona_alignment": dict(state.get("persona_alignment", {})),
            "reputation": dict(state.get("persona_reputation", {})).get(
                npc_name, state.get("reputation", 0.5)
            ),
            "alignment_score": dict(state.get("persona_alignment", {})).get(
                npc_name, state.get("alignment_score", 0.0)
            ),
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
    if not stripped or "\n" in stripped:
        return False

    word_count = len(WORD_RE.findall(stripped))
    has_sentence_punctuation = any(char in stripped for char in ".!?")
    markdown_heading = stripped.startswith("#") or (
        stripped.startswith("**") and stripped.endswith("**")
    )
    title_stub = stripped.endswith(":")
    return word_count <= 14 and not has_sentence_punctuation and (
        markdown_heading or title_stub
    )


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


def _build_turn_focus_instruction(persona: PersonaDefinition, user_text: str) -> str:
    lowered = user_text.lower()
    instructions: list[str] = []
    asks_for_tradeoff = any(
        term in lowered
        for term in ["trade-off", "tradeoff", "balance", "what should we do", "recommendation"]
    )

    if persona.route == "executive" and asks_for_tradeoff:
        instructions.append(
            "Name one concrete decision the business should prioritize and one scope item that should wait."
        )

    if persona.route == "people":
        asks_for_adoption = any(
            term in lowered
            for term in ["adoption", "training", "coaching", "capability", "onboarding"]
        )
        if asks_for_adoption:
            instructions.append(
                "Be specific about how adoption will happen in practice, not just why it matters."
            )
        if asks_for_tradeoff:
            instructions.append(
                "State one trade-off between adoption quality and rollout speed or overhead."
            )

    if persona.route == "operations":
        asks_for_rollout = any(
            term in lowered
            for term in ["rollout", "local", "region", "regional", "staffing", "operations", "implementation"]
        )
        if asks_for_rollout or asks_for_tradeoff:
            instructions.append(
                "Name at least one implementation constraint involving staffing, timing, sequencing, or local readiness."
            )

    return " ".join(instructions)


def _simulation_context_block() -> str:
    criteria = "\n".join(f"- {criterion}" for criterion in ACTIVE_SIMULATION.success_criteria)
    return (
        f"Current simulation: {ACTIVE_SIMULATION.title}.\n"
        f"Brief: {ACTIVE_SIMULATION.brief}\n"
        f"Success criteria:\n{criteria}"
    )


def _meeting_instruction_block(
    *,
    meeting_notes: list[dict],
    last_user_message: str,
    is_final_meeting_response: bool,
) -> str:
    if not meeting_notes and not is_final_meeting_response:
        return (
            "\n\n[MEETING MODE]: You are the first contributor in a visible cross-functional chat. "
            "Give your own perspective only. Do not summarize other contributors and do not present "
            "your message as the final conclusion."
        )

    notes_text = "\n".join(
        f"- {note.get('npc', 'Unknown')}: {note.get('content', '')}" for note in meeting_notes
    ) or "- No prior meeting notes."

    if is_final_meeting_response:
        return (
            "\n\n[MEETING MODE]: Prior contributors have already spoken. Review their notes and "
            "deliver the final visible recommendation in your own voice. Synthesize the prior views, "
            "name at least one trade-off, and make the recommendation feel conclusive.\n\n"
            f"[PRIOR MEETING NOTES]\nUser request: {last_user_message}\n{notes_text}"
        )

    return (
        "\n\n[MEETING MODE]: Other contributors may already have spoken. Add your own perspective "
        "to the visible chat, but do not write the final conclusion.\n\n"
        f"[PRIOR MEETING NOTES]\nUser request: {last_user_message}\n{notes_text}"
    )


def _build_visible_response(
    *, speaker: str, content: str, is_final: bool
) -> VisibleResponse:
    return {
        "speaker": speaker,
        "content": content,
        "is_final": is_final,
    }


def build_npc_node(persona: PersonaDefinition):
    agent_tools = [
        calculate_kpi,
        retrieve_simulation_context,
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
        is_meeting = state.get("mode") == "meeting"
        current_meeting_queue = list(state.get("meeting_queue", []))
        is_final_meeting_response = (
            is_meeting
            and len(current_meeting_queue) == 1
            and current_meeting_queue[0] == persona.route
        )
        soul_markdown, knowledge_markdown = load_persona_memory(persona)
        reputation_state = _update_persona_reputation(
            state, persona.name, last_user_message.lower()
        )
        reputation = reputation_state["reputation"]
        knowledge_chunks = retrieve_knowledge(
            last_user_message, namespaces=[persona.route], top_k=3
        )

        if reputation >= 0.8:
            warmth = "highly trusting and open"
        elif reputation >= 0.5:
            warmth = "neutral and professional"
        else:
            warmth = "skeptical and guarded"

        system_message_content = (
            "Use these markdown files as your durable internal context.\n\n"
            f"[SOUL.md]\n{soul_markdown}\n\n"
            f"[Knowledge.md]\n{knowledge_markdown}\n\n"
            f"Answer carefully as the selected role: {persona.name}.\n\n"
            "Keep the answer natural and human-sized. Default to one short paragraph with 2 to 4 "
            "sentences. Do not write a consultant-style memo. Do not give a multi-step framework "
            "unless the user explicitly asks for one. If the user asks a broad question, give the "
            "single most useful answer first.\n\n"
            f"{_simulation_context_block()}\n\n"
            "Ground your response in the role's priorities and the simulation brief. When the user "
            "asks for a recommendation, framework, or design, state a real trade-off if one matters. "
            "Do not pretend every goal can be maximized at once.\n\n"
            "When the user asks about live tasks, comments, backlog, or status, use the Jira tools "
            "instead of guessing. When the user asks you to create a task, add a comment, or update "
            "a task, use a tool to perform the action. Use your own namespace when calling "
            f"retrieve_simulation_context: `{persona.route}`. Use your own agent_id when calling "
            f"add_jira_comment: `{persona.agent_id}`. After receiving tool results, answer the user "
            "directly and do not mention internal tool mechanics.\n\n"
            f"[RELATIONSHIP CONTEXT]: The user's current reputation with you is "
            f"{reputation:.2f}/1.0 and your attitude is {warmth}. Reflect this in how "
            "forthcoming and warm your response is."
        )

        turn_focus_instruction = _build_turn_focus_instruction(persona, last_user_message)
        if turn_focus_instruction:
            system_message_content += (
                f"\n\n[TURN-SPECIFIC REQUIREMENT]: {turn_focus_instruction}"
            )

        if is_meeting:
            system_message_content += _meeting_instruction_block(
                meeting_notes=list(state.get("meeting_notes", [])),
                last_user_message=last_user_message,
                is_final_meeting_response=is_final_meeting_response,
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

        hint = state.get("supervisor_hint", "")
        if hint:
            system_message_content += (
                f"\n\n[SECRET INSTRUCTION FROM SUPERVISOR]: {hint}"
            )

        system_message = SystemMessage(content=system_message_content)
        windowed_history = full_history[-HISTORY_WINDOW:]
        messages_to_pass = [system_message] + windowed_history

        if tool_enabled_llm is not None:
            response = tool_enabled_llm.invoke(messages_to_pass)
            if getattr(response, "tool_calls", None):
                append_persona_tool_handoff(
                    persona,
                    user_message=last_user_message,
                    tool_names=[
                        str(tool_call.get("name", "tool"))
                        for tool_call in getattr(response, "tool_calls", [])
                    ],
                )
                tool_result = {
                    "messages": [response],
                    "active_npc": persona.name,
                    "visible_responses": list(state.get("visible_responses", [])),
                    "supervisor_hint": state.get("supervisor_hint", ""),
                }
                tool_result.update(reputation_state)
                return tool_result
        else:
            response = llm.invoke(messages_to_pass)

        response_text = (
            response.content if isinstance(response.content, str) else str(response.content)
        )
        response.content = _compress_chat_reply(response_text, last_user_message)
        append_persona_knowledge(
            persona,
            title="Task update",
            user_message=last_user_message,
            agent_response=response.content,
            mode=state.get("mode", "direct_reply"),
        )

        updated_meeting_queue = list(state.get("meeting_queue", []))
        if (
            state.get("mode") == "meeting"
            and updated_meeting_queue
            and updated_meeting_queue[0] == persona.route
        ):
            updated_meeting_queue = updated_meeting_queue[1:]

        updated_meeting_notes = list(state.get("meeting_notes", []))
        updated_visible_responses = list(state.get("visible_responses", []))
        if is_meeting:
            updated_meeting_notes.append(
                {
                    "npc": persona.name,
                    "route": persona.route,
                    "content": response.content,
                }
            )
        updated_visible_responses.append(
            _build_visible_response(
                speaker=persona.name,
                content=response.content,
                is_final=(not is_meeting) or is_final_meeting_response,
            )
        )

        final_state = {
            "messages": [response],
            "active_npc": persona.name,
            "meeting_queue": updated_meeting_queue,
            "meeting_notes": updated_meeting_notes,
            "visible_responses": updated_visible_responses,
            "supervisor_hint": (
                "" if not is_meeting else state.get("supervisor_hint", "")
            ),
        }
        final_state.update(reputation_state)
        return final_state

    return node


persona_nodes = {
    persona.route: build_npc_node(persona) for persona in ACTIVE_SIMULATION.personas
}
