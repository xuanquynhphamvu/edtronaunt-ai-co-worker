from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from .utils.state import AgentState
from .utils.nodes import ceo_node, chro_node, regional_node, safety_node, meeting_synthesis_node
from .agent import supervisor_plan_node
from .utils.tools import (
    add_jira_comment,
    calculate_kpi,
    create_jira_task,
    list_jira_tasks,
    retrieve_brand_data,
    search_jira_tasks,
    update_jira_status,
)

def safety_router(state: AgentState) -> str:
    """After safety_node: if flags were raised, go directly to END; otherwise continue."""
    if state.get("safety_flags"):
        return "end"
    return "supervisor_plan"


def supervisor_router(state: AgentState) -> str:
    mode = state.get("mode", "")
    if mode == "direct_reply":
        route = state.get("target_npc", "end")
        return route if route in ["ceo", "chro", "regional"] else "end"

    meeting_queue = state.get("meeting_queue", [])
    if meeting_queue:
        route = meeting_queue[0]
        return route if route in ["ceo", "chro", "regional"] else "end"
    return "end"


def agent_tools_router(state: AgentState) -> str:
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    if getattr(last_message, "tool_calls", None):
        return "tools"

    if state.get("mode") == "meeting":
        meeting_queue = state.get("meeting_queue", [])
        if meeting_queue:
            route = meeting_queue[0]
            if route in ["ceo", "chro", "regional"]:
                return route
        return "meeting_synthesis"
    return "end"


def return_from_tools_router(state: AgentState) -> str:
    active_npc = state.get("active_npc", "")
    if active_npc == "CEO":
        return "ceo"
    if active_npc == "CHRO":
        return "chro"
    if active_npc == "Regional Manager":
        return "regional"
    return "end"

# 1. Initialize Graph with our Custom TypedDict
workflow = StateGraph(AgentState)

# 2. Add all Nodes
workflow.add_node("safety", safety_node)
workflow.add_node("supervisor_plan", supervisor_plan_node)
workflow.add_node("ceo", ceo_node)
workflow.add_node("chro", chro_node)
workflow.add_node("regional", regional_node)
workflow.add_node("meeting_synthesis", meeting_synthesis_node)
workflow.add_node(
    "tools",
    ToolNode(
        [
            calculate_kpi,
            retrieve_brand_data,
            list_jira_tasks,
            search_jira_tasks,
            create_jira_task,
            add_jira_comment,
            update_jira_status,
        ]
    ),
)

# ── Execution Order ──────────────────────────────────────────
# START → safety → supervisor_plan → (direct NPC | meeting queue) → synthesis/end

# Step 1: Safety check always runs first
workflow.add_edge(START, "safety")

# Step 2: Safety router — short-circuit to END if blocked, else plan the turn
workflow.add_conditional_edges(
    "safety",
    safety_router,
    {
        "supervisor_plan": "supervisor_plan",
        "end": END,
    }
)

# Step 3: Supervisor routes to a direct NPC or starts the meeting queue
workflow.add_conditional_edges(
    "supervisor_plan",
    supervisor_router,
    {
        "ceo": "ceo",
        "chro": "chro",
        "regional": "regional",
        "end": END
    }
)

# Step 4: Agents either call tools, continue the meeting, synthesize, or finish
workflow.add_conditional_edges(
    "ceo",
    agent_tools_router,
    {
        "tools": "tools",
        "ceo": "ceo",
        "chro": "chro",
        "regional": "regional",
        "meeting_synthesis": "meeting_synthesis",
        "end": END,
    }
)
workflow.add_conditional_edges(
    "chro",
    agent_tools_router,
    {
        "tools": "tools",
        "ceo": "ceo",
        "chro": "chro",
        "regional": "regional",
        "meeting_synthesis": "meeting_synthesis",
        "end": END,
    }
)
workflow.add_conditional_edges(
    "regional",
    agent_tools_router,
    {
        "tools": "tools",
        "ceo": "ceo",
        "chro": "chro",
        "regional": "regional",
        "meeting_synthesis": "meeting_synthesis",
        "end": END,
    }
)

# Step 5: Tool results return to the same active NPC for final response
workflow.add_conditional_edges(
    "tools",
    return_from_tools_router,
    {
        "ceo": "ceo",
        "chro": "chro",
        "regional": "regional",
        "end": END,
    }
)

# Step 6: Meeting synthesis ends the turn
workflow.add_edge("meeting_synthesis", END)

engine = workflow.compile()

# Example Usage
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    print("======= CO-WORKER ENGINE =======")
    print(
        "Type your message below (try tagging @CEO, @CHRO, or @regional). Type 'exit' to quit.\n"
    )

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break

            # We only need to pass the new message. MemorySaver handles the rest!
            input_state = {
                "messages": [HumanMessage(content=user_input)]
            }
            config = {"configurable": {"thread_id": "cli_test"}}

            final_state = engine.invoke(input_state, config=config)
            print(f"\n{final_state['messages'][-1].content}")
            print(f"(Handled by: {final_state.get('active_npc', 'System')}) | Turns: {final_state.get('turn_count', 0)}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
