from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import os
from .utils.state import AgentState

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen2.5:32b"), temperature=0.7)

MEETING_QUEUE_DEFAULT = ["ceo", "chro", "regional"]
DIRECT_ROUTES = {"ceo", "chro", "regional"}


class SupervisorPlanOutput(BaseModel):
    mode: str = Field(description="Either 'direct_reply' or 'meeting'.")
    target_npc: str = Field(description="Use 'ceo', 'chro', 'regional', or 'end'. For meeting, return 'end'.")


def _is_broad_meeting_prompt(message_text: str) -> bool:
    lowered = message_text.lower()
    broad_markers = [
        "final recommendation",
        "recommendation",
        "design a",
        "design an",
        "framework",
        "what should we do",
        "leadership system",
        "proposal",
        "plan",
        "roadmap",
        "balance",
        "trade-off",
        "tradeoff",
    ]
    return any(marker in lowered for marker in broad_markers)


def _route_from_explicit_tag(message_text: str) -> str | None:
    lowered = message_text.lower()
    if "@ceo" in lowered:
        return "ceo"
    if "@chro" in lowered:
        return "chro"
    if "@regional" in lowered or "@regional manager" in lowered:
        return "regional"
    return None

def supervisor_plan_node(state: AgentState):
    """Invisible supervisor that decides between direct reply and meeting flow."""
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
        return {
            "active_npc": "Supervisor",
            "mode": "direct_reply",
            "target_npc": explicit_route,
            "next_route": explicit_route,
            "meeting_queue": [],
            "meeting_notes": [],
            "final_response_mode": "supervisor_narrator",
            "reputation_updated_for_turn": [],
            "user_sentiment": "neutral",
            "turn_count": turn_count,
            "supervisor_hint": supervisor_hint,
        }

    if _is_broad_meeting_prompt(user_text):
        return {
            "active_npc": "Supervisor",
            "mode": "meeting",
            "target_npc": "",
            "next_route": MEETING_QUEUE_DEFAULT[0],
            "meeting_queue": list(MEETING_QUEUE_DEFAULT),
            "meeting_notes": [],
            "final_response_mode": "supervisor_narrator",
            "reputation_updated_for_turn": [],
            "user_sentiment": "neutral",
            "turn_count": turn_count,
            "supervisor_hint": supervisor_hint,
        }

    system_msg = SystemMessage(
        content=(
            "You are the invisible Supervisor of the company. "
            "Choose whether the user needs a single direct reply from one coworker or a "
            "cross-functional meeting. Use 'meeting' for broad design, framework, "
            "trade-off, or final recommendation requests. Use 'direct_reply' when one "
            "coworker should answer directly. For direct reply, choose 'ceo', 'chro', "
            "or 'regional'. For meeting, return target_npc='end'."
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
        next_route = meeting_queue[0]
    else:
        meeting_queue = []
        next_route = target_npc

    return {
        "active_npc": "Supervisor",
        "mode": mode,
        "target_npc": target_npc,
        "next_route": next_route,
        "meeting_queue": meeting_queue,
        "meeting_notes": [],
        "final_response_mode": "supervisor_narrator",
        "reputation_updated_for_turn": [],
        "user_sentiment": "neutral",
        "turn_count": turn_count,
        "supervisor_hint": supervisor_hint,
    }
