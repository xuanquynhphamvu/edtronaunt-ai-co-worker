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
    executive_brief: str = Field(
        default="",
        description="If mode is meeting, the CEO's task brief for this meeting.",
    )
    people_brief: str = Field(
        default="",
        description="If mode is meeting, the CHRO's task brief for this meeting.",
    )
    operations_brief: str = Field(
        default="",
        description="If mode is meeting, the operations leader's task brief for this meeting.",
    )
    final_synthesis_goal: str = Field(
        default="",
        description="If mode is meeting, the final decision or synthesis question the meeting should resolve.",
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


def _default_meeting_role_hints(user_text: str) -> dict[str, str]:
    shared_rules = (
        "Speak only from your role's lens. Do not restate the shared brief, problem statement, "
        "or autonomy-versus-capability framing unless you are directly challenging it. Do not open "
        "with agreement. Add one new concern, one challenge or caution, and one recommendation that "
        "is specific to your function."
    )
    return {
        "executive": (
            f"{shared_rules} Focus on the business decision to make now, the boundary to set, what "
            "should wait, and the risk of getting the sequence wrong."
        ),
        "people": (
            f"{shared_rules} Focus on adoption risk, the manager behavior change required, what teams "
            "will need to do differently, and the minimum enablement needed to make the decision stick."
        ),
        "operations": (
            f"{shared_rules} Focus on rollout friction, local variation, communications burden, "
            "timing constraints, and where execution could fail in regions or teams."
        ),
    }


def _coerce_meeting_role_hints(decision: SupervisorPlanOutput, user_text: str) -> dict[str, str]:
    fallback = _default_meeting_role_hints(user_text)
    hints = {
        "executive": str(decision.executive_brief or "").strip(),
        "people": str(decision.people_brief or "").strip(),
        "operations": str(decision.operations_brief or "").strip(),
    }
    final_synthesis_goal = str(decision.final_synthesis_goal or "").strip()
    for route, fallback_hint in fallback.items():
        hint = hints.get(route, "")
        if not hint:
            hints[route] = fallback_hint
            continue
        if final_synthesis_goal:
            hints[route] = f"{hint} Final synthesis target: {final_synthesis_goal}"
    return hints


def _build_supervisor_system_message(supervisor_soul: str, supervisor_knowledge: str) -> SystemMessage:
    return SystemMessage(
        content=(
            "Use these markdown files as the Supervisor's durable internal memory.\n\n"
            f"[SOUL.md]\n{supervisor_soul}\n\n"
            f"[Knowledge.md]\n{supervisor_knowledge}\n\n"
            "You are the invisible Supervisor of the company. Choose whether the user needs a "
            "single direct reply from one coworker or a cross-functional meeting. Use 'meeting' "
            "for broad design, framework, trade-off, or final recommendation requests. Use "
            "'direct_reply' when one coworker should answer directly. Available coworker routes: "
            f"{_route_prompt_fragment()}.\n\n"
            "If you choose meeting, break the task into distinct, non-overlapping role briefs. "
            "Each brief must tell that persona what unique gap to cover, what not to repeat, and "
            "what concrete output to provide. Avoid generic instructions like 'give your perspective'. "
            "For meetings, return target_npc='end'."
        )
    )


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
            "meeting_role_hints": {},
            "meeting_notes": [],
            "visible_responses": [],
            "final_response_mode": "supervisor_narrator",
            "reputation_updated_for_turn": [],
            "user_sentiment": "neutral",
            "turn_count": turn_count,
            "supervisor_hint": supervisor_hint,
        }

    if _is_broad_meeting_prompt(user_text):
        system_msg = _build_supervisor_system_message(supervisor_soul, supervisor_knowledge)
        messages_to_pass = [system_msg] + list(messages)
        planner_llm = llm.with_structured_output(SupervisorPlanOutput)
        try:
            decision = planner_llm.invoke(messages_to_pass)
            meeting_role_hints = _coerce_meeting_role_hints(decision, user_text)
        except Exception:
            meeting_role_hints = _default_meeting_role_hints(user_text)
        append_supervisor_knowledge(
            "Routing decision",
            [
                "Mode: meeting",
                f"User request: {' '.join(user_text.split())[:220]}",
                "Target route: end",
                f"Meeting briefs: {meeting_role_hints}",
            ],
            session_id=session_id,
        )
        return {
            "active_npc": "Supervisor",
            "mode": "meeting",
            "target_npc": "",
            "next_route": MEETING_QUEUE_DEFAULT[0] if MEETING_QUEUE_DEFAULT else "end",
            "meeting_queue": list(MEETING_QUEUE_DEFAULT),
            "meeting_role_hints": meeting_role_hints,
            "meeting_notes": [],
            "visible_responses": [],
            "final_response_mode": "supervisor_narrator",
            "reputation_updated_for_turn": [],
            "user_sentiment": "neutral",
            "turn_count": turn_count,
            "supervisor_hint": supervisor_hint,
        }

    system_msg = _build_supervisor_system_message(supervisor_soul, supervisor_knowledge)
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
        meeting_role_hints = _coerce_meeting_role_hints(decision, user_text)
        next_route = meeting_queue[0] if meeting_queue else "end"
    else:
        meeting_queue = []
        meeting_role_hints = {}
        next_route = target_npc

    append_supervisor_knowledge(
        "Routing decision",
        [
            f"Mode: {mode}",
            f"User request: {' '.join(user_text.split())[:220]}",
            f"Target route: {target_npc or next_route}",
            f"Meeting briefs: {meeting_role_hints or 'none'}",
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
        "meeting_role_hints": meeting_role_hints,
        "meeting_notes": [],
        "visible_responses": [],
        "final_response_mode": "supervisor_narrator",
        "reputation_updated_for_turn": [],
        "user_sentiment": "neutral",
        "turn_count": turn_count,
        "supervisor_hint": supervisor_hint,
    }
