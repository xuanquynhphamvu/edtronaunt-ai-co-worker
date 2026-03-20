from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .agent import supervisor_plan_node
from .simulation import ACTIVE_SIMULATION
from .utils.nodes import ROUTE_BY_NPC, meeting_synthesis_node, persona_nodes, safety_node
from .utils.state import AgentState
from .utils.tools import (
    add_jira_comment,
    calculate_kpi,
    create_jira_task,
    list_jira_tasks,
    retrieve_simulation_context,
    search_jira_tasks,
    update_jira_status,
)

PERSONA_ROUTES = set(ACTIVE_SIMULATION.all_routes)


def safety_router(state: AgentState) -> str:
    if state.get("safety_flags"):
        return "end"
    return "supervisor_plan"


def supervisor_router(state: AgentState) -> str:
    mode = state.get("mode", "")
    if mode == "direct_reply":
        route = state.get("target_npc", "end")
        return route if route in PERSONA_ROUTES else "end"

    meeting_queue = state.get("meeting_queue", [])
    if meeting_queue:
        route = meeting_queue[0]
        return route if route in PERSONA_ROUTES else "end"
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
            if route in PERSONA_ROUTES:
                return route
        return "meeting_synthesis"
    return "end"


def return_from_tools_router(state: AgentState) -> str:
    return ROUTE_BY_NPC.get(state.get("active_npc", ""), "end")


workflow = StateGraph(AgentState)
workflow.add_node("safety", safety_node)
workflow.add_node("supervisor_plan", supervisor_plan_node)
for route, node in persona_nodes.items():
    workflow.add_node(route, node)
workflow.add_node("meeting_synthesis", meeting_synthesis_node)
workflow.add_node(
    "tools",
    ToolNode(
        [
            calculate_kpi,
            retrieve_simulation_context,
            list_jira_tasks,
            search_jira_tasks,
            create_jira_task,
            add_jira_comment,
            update_jira_status,
        ]
    ),
)

workflow.add_edge(START, "safety")
workflow.add_conditional_edges(
    "safety",
    safety_router,
    {
        "supervisor_plan": "supervisor_plan",
        "end": END,
    },
)

supervisor_edges = {route: route for route in PERSONA_ROUTES}
supervisor_edges["end"] = END
workflow.add_conditional_edges("supervisor_plan", supervisor_router, supervisor_edges)

persona_edges = {route: route for route in PERSONA_ROUTES}
persona_edges.update({"tools": "tools", "meeting_synthesis": "meeting_synthesis", "end": END})
for route in PERSONA_ROUTES:
    workflow.add_conditional_edges(route, agent_tools_router, persona_edges)

tool_edges = {route: route for route in PERSONA_ROUTES}
tool_edges["end"] = END
workflow.add_conditional_edges("tools", return_from_tools_router, tool_edges)

workflow.add_edge("meeting_synthesis", END)

engine = workflow.compile()

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    print("======= CO-WORKER ENGINE =======")
    print(
        "Type your message below "
        f"(try tagging {ACTIVE_SIMULATION.tag_hints}). Type 'exit' to quit.\n"
    )

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break

            input_state = {"messages": [HumanMessage(content=user_input)]}
            config = {"configurable": {"thread_id": "cli_test"}}

            final_state = engine.invoke(input_state, config=config)
            print(f"\n{final_state['messages'][-1].content}")
            print(
                f"(Handled by: {final_state.get('active_npc', 'System')}) | "
                f"Turns: {final_state.get('turn_count', 0)}\n"
            )
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
