import sys
import os
import json
from llm_inference.llm_inference_base import ai_completion, ai_completion_with_backoff, truncate_text, get_length
from llm_inference.llm_response_parser import parse_llm_response
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False


def reset_chat():
    """Resets the chat and configuration settings."""
    st.session_state.confirm_reset = False
    st.rerun()


def abort_reset():
    st.session_state.confirm_reset = False


if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for system prompt
if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = "You are a helful assistant."

def on_model_change():
    st.session_state.messages = []

with st.sidebar:
    st.title("Chat app")
    # Model selection dropdown
    model_options = ["gemini-2.0-flash", "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-05-06"]
    st.session_state["model"] = st.selectbox(
        "Select Model",
        options=model_options,
        index=0,
        on_change=on_model_change,
    )
    st.session_state["system_prompt"] = st.text_area(
        "System Prompt",
        value=st.session_state["system_prompt"],
        height=100,
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Input your message here"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        oai_messages = [{"role": "system", "content": st.session_state["system_prompt"]}] + st.session_state.messages
        print(oai_messages)
        response = ai_completion(
            model=st.session_state["model"],
            messages=oai_messages,
            stream=True,
        )
        response = st.write_stream(response["content"])

    st.session_state.messages.append({"role": "assistant", "content": response})



# "Start Over" Button with Confirmation
if not st.session_state.confirm_reset:
    if st.sidebar.button("Start Over"):
        st.session_state.confirm_reset = True
        st.rerun()
else:
    st.sidebar.warning("Are you sure you want to restart? This will clear the chat and configurations.")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("Yes, Restart"):
            reset_chat()
    with col2:
        if st.sidebar.button("No, Cancel"):
            abort_reset()