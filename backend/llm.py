import os
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import create_qa_with_sources_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from backend.vector_loader import load_chroma_db 
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
LM_MODEL_NAME = os.getenv("MODEL_NAME")


# Disable parallelism to avoid tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENAI_API_KEY"] = API_KEY

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based only on the provided context.
Your goal is to generate responses for the user question based on the information given.
Instructions:
1. Answer the question only based on given context.
2. If you don't know the answer, simply say you don't know. Never make up answers.
3. Maintain the context of the conversation. Do not provide irrelevant information.
4. If you can find the answer try to add as much detail from the context.
5. Do not answer the question based on assumptions using model internal memory.

"""

USER_PROMPT = """
Conversation history:
{history}

User question:
{question}

Relevant documents:
{context}

Please provide an answer based on the above information.
"""

# Function to create conversational retrieval system
def create_retrieval_qa_system(user_email, selected_document):
    vector_db = load_chroma_db()
    user_email_filter = {"user_email": user_email}
    pdf_filename_filter = {"pdf_filename": selected_document}

    metadata_filter = {
    "$and": [
        user_email_filter,
        pdf_filename_filter
    ]
    }
    retriever = vector_db.as_retriever(filter = metadata_filter, search_kwargs={"k": 5})
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    llm = ChatOpenAI(model=LM_MODEL_NAME, temperature=0)
    
    chat_prompt = ChatPromptTemplate.from_template(USER_PROMPT)
    qa = create_qa_with_sources_chain(llm, prompt=chat_prompt)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    return qa_chain

# Function to ask a question
def ask_question(qa_chain, question, chat_history):
    response = qa_chain.invoke({"question": question, "chat_history": chat_history})
    
    # If the response is "I don't know", try searching tables explicitly
    if response.get('answer', "").strip().lower() == "i don't know":
        print("Re-trying query in tables only...")
        vector_db = load_chroma_db()
        table_results = vector_db.similarity_search(question, filter={"type": "table"})
        
        if table_results:
            table_data = "\n\n".join([doc.page_content for doc in table_results])
            return {"answer": f"Here are relevant table results:\n\n{table_data}"}
    

    return response