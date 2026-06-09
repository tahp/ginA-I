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

    # Model Management
    with st.expander("🛠️ Manage Models"):
        # Pull Model
        new_model = st.text_input("Pull New Model", placeholder="e.g., llama3")
        if st.button("Download Model", use_container_width=True):
            if new_model:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                async def do_pull():
                    try:
                        success = False
                        async for part in st.session_state.client.pull_model(new_model):
                            status = part.get('status', '')
                            completed = part.get('completed', 0)
                            total = part.get('total', 0)
                            
                            if status == 'success':
                                success = True
                            
                            if total > 0:
                                progress = completed / total
                                progress_bar.progress(progress)
                                status_text.text(f"{status}: {int(progress * 100)}%")
                            else:
                                status_text.text(status)
                        
                        if success:
                            st.success(f"Model {new_model} pulled successfully!")
                            st.rerun()
                        else:
                            st.error(f"Failed to pull model {new_model}. Check if the name is correct.")
                    except Exception as e:
                        if e.__class__.__name__ == 'StopException':
                            raise
                        st.error(f"Error during pull: {str(e)}")

                asyncio.run(do_pull())
            else:
                st.warning("Please enter a model name.")

        st.divider()

        # Delete Model
        if available_models:
            model_to_delete = st.selectbox("Select Model to Delete", options=available_models)
            if st.button("🗑️ Delete Model", use_container_width=True, type="secondary"):
                if st.session_state.config.model_name == model_to_delete:
                    st.error("Cannot delete the currently active model.")
                else:
                    if asyncio.run(st.session_state.client.delete_model(model_to_delete)):
                        st.success(f"Model {model_to_delete} deleted.")
                        st.rerun()
                    else:
                        st.error(f"Failed to delete {model_to_delete}.")

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
        
        async def stream_response():
            full_response = ""
            try:
                async for chunk in st.session_state.client.get_response(prompt):
                    # Handle system/tool call notifications visually
                    if "[System:" in chunk:
                        st.info(chunk.strip("[]"))
                    else:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            except Exception as e:
                # If it's a Streamlit StopException, let it bubble up
                if e.__class__.__name__ == 'StopException':
                    raise
                st.error(f"An error occurred: {str(e)}")
                # Re-raise if it's not a common error we want to swallow
                if not isinstance(e, (asyncio.CancelledError,)):
                    raise e

        try:
            asyncio.run(stream_response())
        except Exception as e:
            # Most errors are already handled in stream_response or by Streamlit
            pass
