import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# Initialize the Gemini model
# Note: Ensure GOOGLE_API_KEY is in your .env file
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

def get_chat_response(messages):
    """
    Sends the message history to Gemini and returns the response.
    """
    system_instruction = SystemMessage(content="You are a helpful and friendly AI assistant.")
    
    # Prepend system instruction to the history
    full_history = [system_instruction] + messages
    
    response = llm.invoke(full_history)
    return response.content
