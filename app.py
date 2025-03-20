import torch
torch.classes.__path__ = []


from langchain.memory import ConversationBufferMemory
import streamlit as st
from backend.document_loader_processor import USER_ACCESS
from backend.vector_loader import load_chroma_db ,query_chroma_db
from backend.llm import create_retrieval_qa_system, ask_question
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Streamlit UI
st.title("Multi User Document Search")

if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "selected_document" not in st.session_state:
    st.session_state.selected_document = None


if "current_user" not in st.session_state:
    email = st.text_input("Enter your email address:", key="email_input")
    if st.button("Login") and email:
        if email in USER_ACCESS:
            st.session_state.user_email = email
            st.session_state.user_documents = USER_ACCESS[email]
            st.success(f"Welcome {email}!")
            st.rerun()
        else:
            st.error("Invalid email. Please try again.")

if st.button("Logout"):
    st.session_state.user_email = None
    st.session_state.conversation_history = []
    st.rerun()

# Step 2: Show Documents
if st.session_state.user_email:
    st.subheader("Select a Document")

    if st.session_state.user_documents:
        selected_doc = st.selectbox("Available Documents:", st.session_state.user_documents)
        # Reset chat history when the document changes
        if selected_doc != st.session_state.selected_document:
            st.session_state.selected_document = selected_doc
            st.session_state.conversation_history = []  # Reset conversation history when changing document
            st.session_state.chat_memory = {}  # Reset chat memory for all users
            st.success(f"Selected Document: {selected_doc}")

        # if selected_doc:
        #     st.session_state.selected_document = selected_doc
        #     st.success(f"Selected Document: {selected_doc}")
    else:
        st.warning("No documents available for your account.")


if st.session_state.user_email and "selected_document" in st.session_state:
    st.subheader("Chat")
    user_input = st.chat_input("Ask a question:")
    qa_chain = create_retrieval_qa_system(st.session_state.user_email, st.session_state.selected_document)
    
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if user_input:
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        response = ask_question(qa_chain, user_input, st.session_state.conversation_history)
        vector_db = load_chroma_db()
        st.session_state.conversation_history.append({"role": "assistant", "content": response['answer']})
        st.write(response['answer'])

        st.rerun()
