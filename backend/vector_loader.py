
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Assuming the Chroma DB is already loaded
def load_chroma_db(persist_directory="chroma_db"):
    """Load the ChromaDB vector store and check stored documents."""
    db = Chroma(persist_directory=persist_directory, 
                embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    return db

def query_chroma_db(user_email,pdf_filename, user_question, vector_db, top_k=3):
    """Query ChromaDB with user email, selected PDF file, and a given question."""
 
    # # Filter by user email and PDF filename (metadata)

    user_email_filter = {"user_email": user_email}
    pdf_filename_filter = {"pdf_filename": pdf_filename}

    metadata_filter = {
    "$and": [
        user_email_filter,
        pdf_filename_filter
    ]
    }
    
    # Retrieve the relevant documents based on the filtered metadata
    relevant_chunks = vector_db.similarity_search_with_score(user_question,
                                                            k=top_k
                                                            , filter=metadata_filter
                                                            )

    # Return the top-k relevant chunks and their similarity scores
    return relevant_chunks
