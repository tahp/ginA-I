import streamlit as st
import asyncio
import os
from chatbot import ChatClient, ChatConfig

# Set page config
st.set_page_config(page_title="Ollama Chatbot", page_icon="🤖", layout="centered")

# Initialize ChatClient in session state
if "client" not in st.session_state:
    config = ChatConfig()
    st.session_state.client = ChatClient(config)
    st.session_state.config = config

# Sidebar for configuration
with st.sidebar:
    st.title("Settings")
    
    # Model Selection
    available_models = asyncio.run(st.session_state.client.list_models())
    if available_models:
        selected_model = st.selectbox(
            "Select Model", 
            options=available_models, 
            index=available_models.index(st.session_state.config.model_name) if st.session_state.config.model_name in available_models else 0
        )
        if selected_model != st.session_state.config.model_name:
            st.session_state.client.set_model(selected_model)
            st.rerun()
    else:
        st.error("No models found. Ensure Ollama is running.")

    st.divider()
    st.write(f"**Current Model:** {st.session_state.config.model_name}")
    st.write(f"**History Limit:** {st.session_state.config.history_limit}")
    
    if st.button("Clear History"):
        st.session_state.client.history = []
        if st.session_state.config.system_prompt:
            st.session_state.client.history.append({'role': 'system', 'content': st.session_state.config.system_prompt})
        st.session_state.client._save_history()
        st.rerun()

# Main Chat Interface
st.title("🤖 Ollama Web Chat")

# Display chat history (filtering out system messages and tool calls for UI)
for message in st.session_state.client.history:
    if message['role'] in ['user', 'assistant']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

# Chat Input
if prompt := st.chat_input("What is on your mind?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response with streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        async def stream_response():
            nonlocal full_response
            async for chunk in st.session_state.client.get_response(prompt):
                # Handle system/tool call notifications visually
                if "[System:" in chunk:
                    st.info(chunk.strip("[]"))
                else:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

        asyncio.run(stream_response())
