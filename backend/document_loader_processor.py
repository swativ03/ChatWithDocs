
import pdfplumber
import shutil
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document


import os

USER_ACCESS = {
    "alice@email.com": ["Accenture"],
    "bob@email.com": ["Amazon", "CocaCola"],
    "charlie@email.com": ["JPMC", "Walt Disney"]
}
def load_pdf(pdf_path):
    """Loads a PDF file and extracts text, including tables."""
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    # Extract tables using pdfplumber
    extracted_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            
            for table in tables:
                if not table:  # Skip empty tables
                    continue
                
                # Filter out invalid tables: Ensure table has meaningful data
                if all(all(cell is None or str(cell).strip() == "" for cell in row) for row in table):
                    continue  # Skip tables that are completely empty or whitespace
                
                # Normalize table by converting None values to empty strings
                clean_table = [[str(cell).strip() if cell else "" for cell in row] for row in table]
                
                # Flatten tables into a readable string format
                table_str = "\n".join([" | ".join(row) for row in clean_table])
                
                extracted_tables.append((page_num, table_str))  # Store table with page number
    

    return documents, extracted_tables

def format_tables_as_text(tables):
    """Ensures extracted tables are formatted in a structured way"""
    formatted_tables = []
    for table in tables:
        if not table:  
            continue  # Skip empty tables

        formatted_rows = []
        
        for row in table:
            # Ensure row is a list; if it's not, convert it into a single-item list
            if not isinstance(row, list):
                row = [row]  # Convert non-list values (like int) to a list
            
            # Convert each cell to a string and join with " | "
            formatted_row = " | ".join(str(cell) if cell is not None else "N/A" for cell in row)
            formatted_rows.append(formatted_row)

        table_str = "\n".join(formatted_rows)
        
        formatted_tables.append(table_str) 
    return formatted_tables


def chunk_documents_with_metadata(documents, tables, pdf_filename):
    """Splits the documents into chunks with user access metadata."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = []
    
    for document in documents:
        # Split the document into chunks
        split_chunks = text_splitter.split_documents([document])
        
        for chunk in split_chunks:
            # Extract the user's email based on the content or any criteria you choose
            for user_email, companies in USER_ACCESS.items():
                if any(company in chunk.page_content for company in companies):
                    # Attach metadata to each chunk
                    chunk.metadata = {
                        "pdf_filename": pdf_filename,
                        "user_email": user_email
                    }
                    chunks.append(chunk)
        
    # Process table data (if available)
    if tables:
        print("tables:",tables)
        formatted_tables = format_tables_as_text(tables)
        for table_text in formatted_tables:
            table_doc = Document(page_content=table_text, 
                                 metadata={"pdf_filename": pdf_filename, 
                                           "user_email": user_email,
                                           "type": "table"})
            chunks.append(table_doc)
    
    return chunks

def store_in_chroma_with_metadata(chunks, persist_directory="chroma_db"):
    """Embeds document chunks with metadata and stores them in ChromaDB."""
    # Delete existing database directory if it exists
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)  # Removes the entire directory and its contents
        print(f"Deleted existing ChromaDB at {persist_directory}, creating a new one.")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, 
                                         embedding=embeddings, 
                                         persist_directory=persist_directory)
    return vector_store

def process_documents(pdf_paths):
    """Process multiple PDFs, chunk them, add metadata and store in Chroma."""
    all_chunks = []
    print(f"paths: {pdf_paths}")
    for pdf_path in pdf_paths:
        print(f"Processing document: {pdf_path}")
        documents,tables = load_pdf(pdf_path)
        pdf_filename = pdf_path.split("/")[-1]  # Extract the PDF filename
        chunks = chunk_documents_with_metadata(documents, tables, pdf_filename)
        all_chunks.extend(chunks)
    
    # Now store all the chunks in Chroma with embeddings
    vector_db = store_in_chroma_with_metadata(all_chunks)
    return vector_db



if __name__ == "__main__":
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))  # Get current file path (backend folder)
    DOCS_PATH = os.path.join(BASE_PATH, 'docs/')

    pdf_files = [os.path.join(DOCS_PATH, f) for f in os.listdir(DOCS_PATH)]
    # print(pdf_files)
    vector_db = process_documents(pdf_files)
    print("Documents processed and stored in ChromaDB!")
