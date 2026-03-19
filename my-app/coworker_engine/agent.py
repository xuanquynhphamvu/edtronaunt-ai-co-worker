from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from .utils.state import AgentState

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

class RouterOutput(BaseModel):
    next_route: str = Field(description="The next agent to route to: 'ceo', 'chro', 'regional', or 'end' if everything is handled.")

def supervisor_node(state: AgentState):
    """The Director Agent that routes invisibly."""
    
    # --- SAFETY CHECK TO PREVENT INFINITE LOOPS ---
    messages = state.get('messages', [])
    if messages and messages[-1].type == "ai":
        # The preceding message was an Agent replying. 
        # Force the graph to 'end' so the user can read it!
        return {
            "active_npc": "Supervisor",
            "next_route": "end",
            "user_sentiment": "neutral"
        }
    # ----------------------------------------------
    
    turn_count = state.get('turn_count', 0)
    if messages and messages[-1].type == "human":
        turn_count += 1
        
    supervisor_hint = state.get("supervisor_hint", "")
    if turn_count > 3 and not supervisor_hint:
        supervisor_hint = "System Note: The user seems to be stuck or going in circles. Please provide a direct, helpful hint to guide them to the right solution."
    
    system_msg = SystemMessage(
        content="You are the invisible Supervisor of the company. Look at the messages. "
                "If the user is asking a question or tagging an employee, decide who should answer it ('ceo', 'chro', 'regional'). "
                "If an employee just answered, you should ALWAYS route to 'end' to give the turn back to the user."
    )
    
    messages_to_pass = [system_msg] + list(state.get('messages', []))
    router_llm = llm.with_structured_output(RouterOutput)
    decision = router_llm.invoke(messages_to_pass)
    
    # Notice we DO NOT append an AIMessage here. The supervisor is invisible!
    return {
        "active_npc": "Supervisor",
        "next_route": decision.next_route,
        "user_sentiment": "neutral",
        "turn_count": turn_count,
        "supervisor_hint": supervisor_hint
    }
