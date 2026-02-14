import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from backend import get_chat_response

st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖")

st.title("🤖 Simple Gemini Chat")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat history
for message in st.session_state["chat_history"]:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Chat Input
user_input = st.chat_input("Type your message here...")

if user_input:
    # 1. Display and store user message
    st.session_state["chat_history"].append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Get and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Pass the history to the backend
                response_text = get_chat_response(st.session_state["chat_history"])
                st.markdown(response_text)
                
                # Store AI message
                st.session_state["chat_history"].append(AIMessage(content=response_text))
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Sidebar for controls
with st.sidebar:
    if st.button("Clear Conversation"):
        st.session_state["chat_history"] = []
        st.rerun()
