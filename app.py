import streamlit as st
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import sys
import os
from pathlib import Path
from uuid import uuid4

load_dotenv()

# Add my-app folder to sys path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "my-app"))
from coworker_engine.engine import engine
from coworker_engine.utils.portfolio import (
    PORTFOLIO_EXPORT_SESSION_KEY,
    PortfolioError,
    export_portfolio_pack,
    get_portfolio_status,
    save_portfolio_artifact,
)

CASE_BRIEF = """
Design a group-level leadership system for Gucci Group that preserves brand DNA, improves talent development
and inter-brand mobility, and uses 360 feedback plus coaching to grow leaders. You are working inside a
simulated cross-functional meeting with three AI stakeholders who will challenge the trade-off between brand
identity, development impact, and regional rollout realism in your final recommendation.
"""

SUCCESS_CRITERIA = [
    "Protect Gucci Group DNA while still enabling shared leadership language.",
    "Show how the competency framework, 360 feedback, and coaching improve talent development and mobility.",
    "Address regional rollout friction, stakeholder buy-in, and execution overhead.",
    "Deliver a final recommendation that clearly balances brand DNA, development goals, and rollout realism.",
]

EXPORT_ROOT = Path(__file__).resolve().parent / "exports" / "portfolio-packs"


def _flash(level: str, text: str) -> None:
    st.session_state["portfolio_flash"] = {"level": level, "text": text}


def _append_message(role: str, content: str) -> None:
    st.session_state.messages.append(
        {
            "id": str(uuid4()),
            "role": role,
            "content": content,
        }
    )


def _message_id(message: dict, index: int) -> str:
    message_id = message.get("id")
    if isinstance(message_id, str) and message_id.strip():
        return message_id
    return f"legacy-{index}"


def _save_message_as_artifact(
    *,
    artifact_type: str,
    body_markdown: str,
    title: str,
    source_notes: str,
    metadata: dict[str, str] | None = None,
) -> None:
    artifact = save_portfolio_artifact(
        st.session_state,
        artifact_type=artifact_type,
        body_markdown=body_markdown,
        title=title.strip() or None,
        source_notes=source_notes,
        metadata=metadata,
    )
    _flash("success", f"Saved '{artifact.title}' to the Portfolio Pack.")
    st.rerun()


def _render_save_actions(message: dict, message_id: str) -> None:
    body_markdown = message.get("content", "")
    with st.expander("Portfolio capture", expanded=False):
        final_tab, comm_tab, exec_tab = st.tabs(
            ["Final plan", "Internal comm", "Executive update"]
        )

        with final_tab:
            with st.form(f"save-final-plan-{message_id}"):
                title = st.text_input(
                    "Title",
                    value="Final plan",
                    key=f"final-plan-title-{message_id}",
                )
                source_notes = st.text_area(
                    "Additional source notes",
                    key=f"final-plan-sources-{message_id}",
                    help="Optional notes or references to attach to the export.",
                )
                submitted = st.form_submit_button("Save as final plan")
                if submitted:
                    _save_message_as_artifact(
                        artifact_type="final_plan",
                        body_markdown=body_markdown,
                        title=title,
                        source_notes=source_notes,
                    )

        with comm_tab:
            with st.form(f"save-internal-comm-{message_id}"):
                comm_type = st.selectbox(
                    "Communication type",
                    options=["email", "post"],
                    key=f"internal-comm-type-{message_id}",
                )
                title = st.text_input(
                    "Title",
                    value="",
                    placeholder="Optional: a concise subject line or post title",
                    key=f"internal-comm-title-{message_id}",
                )
                audience = st.text_input(
                    "Audience",
                    value="Internal stakeholders",
                    key=f"internal-comm-audience-{message_id}",
                )
                source_notes = st.text_area(
                    "Additional source notes",
                    key=f"internal-comm-sources-{message_id}",
                )
                submitted = st.form_submit_button("Save as internal comm")
                if submitted:
                    _save_message_as_artifact(
                        artifact_type="internal_comm",
                        body_markdown=body_markdown,
                        title=title,
                        source_notes=source_notes,
                        metadata={
                            "comm_type": comm_type,
                            "audience": audience.strip() or "Internal stakeholders",
                        },
                    )

        with exec_tab:
            with st.form(f"save-exec-update-{message_id}"):
                title = st.text_input(
                    "Title",
                    value="Executive update",
                    key=f"exec-update-title-{message_id}",
                )
                audience = st.text_input(
                    "Audience",
                    value="Executive leadership",
                    key=f"exec-update-audience-{message_id}",
                )
                source_notes = st.text_area(
                    "Additional source notes",
                    key=f"exec-update-sources-{message_id}",
                )
                submitted = st.form_submit_button("Save as executive update")
                if submitted:
                    _save_message_as_artifact(
                        artifact_type="executive_update",
                        body_markdown=body_markdown,
                        title=title,
                        source_notes=source_notes,
                        metadata={"audience": audience.strip() or "Executive leadership"},
                    )


def _render_portfolio_sidebar() -> None:
    status = get_portfolio_status(st.session_state)
    flash = st.session_state.get("portfolio_flash")

    st.markdown("### Portfolio Pack")
    if flash:
        level = flash.get("level", "info")
        text = flash.get("text", "")
        if level == "success":
            st.success(text)
        elif level == "error":
            st.error(text)
        else:
            st.info(text)
        st.session_state.pop("portfolio_flash", None)

    counts = status["counts"]
    st.caption(
        "Ready when at least one final plan, one internal communication, and one executive update are saved."
    )
    st.markdown(f"- Final plan: {'Saved' if counts['final_plan'] else 'Missing'}")
    st.markdown(f"- Internal comms: {counts['internal_comm']}")
    st.markdown(f"- Executive update: {'Saved' if counts['executive_update'] else 'Missing'}")

    titles = status["titles"]
    if titles["final_plan"]:
        st.caption(f"Latest plan: {titles['final_plan']}")
    if titles["internal_comm"]:
        st.caption("Internal comms: " + ", ".join(titles["internal_comm"]))
    if titles["executive_update"]:
        st.caption(f"Latest exec update: {titles['executive_update']}")

    export_requested = st.button(
        "Export Portfolio Pack",
        disabled=not status["is_ready"],
        use_container_width=True,
    )

    if export_requested:
        try:
            exported = export_portfolio_pack(
                st.session_state,
                thread_id=st.session_state.thread_id,
                export_root=EXPORT_ROOT,
            )
        except (PortfolioError, RuntimeError) as exc:
            _flash("error", str(exc))
        else:
            _flash("success", f"Portfolio Pack saved to {exported.file_path}.")
        st.rerun()

    if not status["is_ready"]:
        missing = ", ".join(name.replace("_", " ") for name in status["missing"])
        st.caption(f"Export is blocked until these are saved: {missing}.")

    last_export = st.session_state.get(PORTFOLIO_EXPORT_SESSION_KEY)
    if last_export:
        st.download_button(
            "Download Portfolio Pack PDF",
            data=last_export["pdf_bytes"],
            file_name=last_export["file_name"],
            mime="application/pdf",
            use_container_width=True,
        )
        st.caption(f"Workspace copy: {last_export['file_path']}")


st.title("AI Co-worker Engine (LangGraph + Multi-Agent)")
st.markdown(CASE_BRIEF)

# Chat history initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit-{uuid4()}"

with st.sidebar:
    st.markdown("### Success Criteria")
    for item in SUCCESS_CRITERIA:
        st.markdown(f"- {item}")
    _render_portfolio_sidebar()

# Display chat history
for index, message in enumerate(st.session_state.messages):
    message_id = _message_id(message, index)
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_save_actions(message, message_id)

# User input
if prompt := st.chat_input("Ask @CEO, @CHRO, or @regional..."):
    # Append user message
    _append_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke the LangGraph Engine
    with st.chat_message("assistant"):
        input_state = {
            "messages": [HumanMessage(content=prompt)]
        }
        
        # Configure thread for LangGraph memory
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # Invoke Langgraph
        final_state = engine.invoke(input_state, config=config)
        
        # Extract the node and message generated
        final_message = final_state['messages'][-1].content
        active_npc = final_state.get('active_npc', 'System')
        
        response_text = f"{final_message}"
        st.markdown(response_text)

    # Save to history
    _append_message("assistant", response_text)
    st.rerun()
