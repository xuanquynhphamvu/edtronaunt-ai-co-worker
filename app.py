import streamlit as st
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import sys
import os

load_dotenv()

# Add my-app folder to sys path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "my-app"))
from coworker_engine.engine import engine

st.title("AI Co-worker Engine (LangGraph + Multi-Agent)")

# Chat history initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask @CEO, @CHRO, or @regional..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke the LangGraph Engine
    with st.chat_message("assistant"):
        input_state = {
            "messages": [HumanMessage(content=prompt)]
        }
        
        # Configure thread for LangGraph memory
        config = {"configurable": {"thread_id": "streamlit_user_1"}}
        
        # Invoke Langgraph
        final_state = engine.invoke(input_state, config=config)
        
        # Extract the node and message generated
        final_message = final_state['messages'][-1].content
        active_npc = final_state.get('active_npc', 'System')
        
        response_text = f"{final_message}"
        st.markdown(response_text)

    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
