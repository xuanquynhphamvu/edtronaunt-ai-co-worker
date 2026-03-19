from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Tracks the conversation history; new messages are appended
    messages: Annotated[list[BaseMessage], add_messages]
    user_sentiment: str
    active_npc: str
    task_progress: str
    next_route: str
    turn_count: int
    user_frustration_level: int
    supervisor_hint: str
    
    # --- New Gamified NPC Mechanics ---
    reputation: float          # 0.0 (Skeptical) to 1.0 (Trusting)
    alignment_score: float     # Tracks how closely the user aligns with brand values
    session_id: str            # Unique ID for the current interaction session
    safety_flags: list[str]    # Flags caught by the pre-LLM safety check
