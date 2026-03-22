from __future__ import annotations

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from .simulation import ACTIVE_SIMULATION
from .utils.agent_memory import (
    append_supervisor_knowledge,
    ensure_simulation_agent_files,
    load_supervisor_memory,
)
from .utils.model_provider import create_chat_model
from .utils.state import AgentState

llm = create_chat_model(temperature=0.7)

MEETING_QUEUE_DEFAULT = list(ACTIVE_SIMULATION.all_routes)
DIRECT_ROUTES = set(ACTIVE_SIMULATION.all_routes)
PERSONA_BY_ROUTE = {persona.route: persona for persona in ACTIVE_SIMULATION.personas}
ensure_simulation_agent_files(ACTIVE_SIMULATION)


class SupervisorPlanOutput(BaseModel):
    mode: str = Field(description="Either 'direct_reply' or 'meeting'.")
    target_npc: str = Field(
        description="Return one configured persona route for direct replies, or 'end' for meetings."
    )


def _is_broad_meeting_prompt(message_text: str) -> bool:
    lowered = message_text.lower()
    broad_markers = [
        "final recommendation",
        "recommendation",
        "design a",
        "design an",
        "framework",
        "what should we do",
        "proposal",
        "plan",
        "roadmap",
        "balance",
        "trade-off",
        "tradeoff",
        "operating model",
        "rollout plan",
    ]
    return any(marker in lowered for marker in broad_markers)


def _route_from_explicit_tag(message_text: str) -> str | None:
    lowered = message_text.lower()
    for route, persona in PERSONA_BY_ROUTE.items():
        if any(alias.lower() in lowered for alias in persona.aliases):
            return route
    return None


def _route_prompt_fragment() -> str:
    fragments = [
        f"{persona.route} ({persona.name}; tags: {', '.join(persona.aliases)})"
        for persona in ACTIVE_SIMULATION.personas
    ]
    return "; ".join(fragments)


def supervisor_plan_node(state: AgentState):
    session_id = str(state.get("session_id", "")).strip() or None
    supervisor_soul, supervisor_knowledge = load_supervisor_memory(session_id=session_id)
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    user_text = last_message.content if last_message and last_message.type == "human" else ""
    explicit_route = _route_from_explicit_tag(user_text)

    turn_count = state.get("turn_count", 0)
    if last_message and last_message.type == "human":
        turn_count += 1

    supervisor_hint = state.get("supervisor_hint", "")
    if turn_count > 3 and not supervisor_hint:
        supervisor_hint = (
            "System Note: The user seems to be stuck or going in circles. "
            "Please provide a direct, helpful hint to guide them to the right solution."
        )

    if explicit_route:
        append_supervisor_knowledge(
            "Routing decision",
            [
                f"Mode: direct_reply",
                f"User request: {' '.join(user_text.split())[:220]}",
                f"Target route: {explicit_route}",
            ],
            session_id=session_id,
        )
        return {
            "active_npc": "Supervisor",
            "mode": "direct_reply",
            "target_npc": explicit_route,
            "next_route": explicit_route,
            "meeting_queue": [],
            "meeting_notes": [],
            "visible_responses": [],
            "final_response_mode": "supervisor_narrator",
            "reputation_updated_for_turn": [],
            "user_sentiment": "neutral",
            "turn_count": turn_count,
            "supervisor_hint": supervisor_hint,
        }

    if _is_broad_meeting_prompt(user_text):
        append_supervisor_knowledge(
            "Routing decision",
            [
                "Mode: meeting",
                f"User request: {' '.join(user_text.split())[:220]}",
                "Target route: end",
            ],
            session_id=session_id,
        )
        return {
            "active_npc": "Supervisor",
            "mode": "meeting",
            "target_npc": "",
            "next_route": MEETING_QUEUE_DEFAULT[0] if MEETING_QUEUE_DEFAULT else "end",
            "meeting_queue": list(MEETING_QUEUE_DEFAULT),
            "meeting_notes": [],
            "visible_responses": [],
            "final_response_mode": "supervisor_narrator",
            "reputation_updated_for_turn": [],
            "user_sentiment": "neutral",
            "turn_count": turn_count,
            "supervisor_hint": supervisor_hint,
        }

    system_msg = SystemMessage(
        content=(
            "Use these markdown files as the Supervisor's durable internal memory.\n\n"
            f"[SOUL.md]\n{supervisor_soul}\n\n"
            f"[Knowledge.md]\n{supervisor_knowledge}\n\n"
            "You are the invisible Supervisor of the company. Choose whether the user needs a "
            "single direct reply from one coworker or a cross-functional meeting. Use 'meeting' "
            "for broad design, framework, trade-off, or final recommendation requests. Use "
            "'direct_reply' when one coworker should answer directly. Available coworker routes: "
            f"{_route_prompt_fragment()}. For meetings, return target_npc='end'."
        )
    )
    messages_to_pass = [system_msg] + list(messages)
    router_llm = llm.with_structured_output(SupervisorPlanOutput)
    decision = router_llm.invoke(messages_to_pass)

    mode = decision.mode if decision.mode in {"direct_reply", "meeting"} else "meeting"
    target_npc = decision.target_npc if decision.target_npc in DIRECT_ROUTES else ""
    if mode == "direct_reply" and not target_npc:
        mode = "meeting"

    if mode == "meeting":
        target_npc = ""
        meeting_queue = list(MEETING_QUEUE_DEFAULT)
        next_route = meeting_queue[0] if meeting_queue else "end"
    else:
        meeting_queue = []
        next_route = target_npc

    append_supervisor_knowledge(
        "Routing decision",
        [
            f"Mode: {mode}",
            f"User request: {' '.join(user_text.split())[:220]}",
            f"Target route: {target_npc or next_route}",
            f"Supervisor hint: {supervisor_hint or 'none'}",
        ],
        session_id=session_id,
    )

    return {
        "active_npc": "Supervisor",
        "mode": mode,
        "target_npc": target_npc,
        "next_route": next_route,
        "meeting_queue": meeting_queue,
        "meeting_notes": [],
        "visible_responses": [],
        "final_response_mode": "supervisor_narrator",
        "reputation_updated_for_turn": [],
        "user_sentiment": "neutral",
        "turn_count": turn_count,
        "supervisor_hint": supervisor_hint,
    }
