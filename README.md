# ChatWithDocs

## Multi-User Document Search Streamlit App  

This project is a multi-user document search and Q&A system that enables users to query their uploaded documents and retrieve relevant information efficiently. The system ensures user-specific access control, allowing each user to access only their authorized documents.
DEMO : [![Watch the demo](https://img.shields.io/badge/Watch-Demo-blue?style=for-the-badge)](https://drive.google.com/file/d/1t2P6kQZ0iMOr-FLByt7QJpoiS6oY36xV/view?usp=sharing)

## Features
- Secure login system to maintain user-specific document access.
- Multi-document support, resetting chat history when switching documents.
- Context-aware retrieval using LangChain and ChromaDB for accurate responses.
- Conversational Q&A experience with memory retention for ongoing discussions.
- Streamlit-based UI for a seamless and interactive experience.

## Prerequisites
- Python 3.8+
- OpenAI API key (required for querying the AI model)

## Installation and Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

2. **Create a virtual environment (optional but recommended)**
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies**
    pip install -r requirements.txt


4. **Set up environment variables**
    - Rename .env.example to .env
    - Add your OPENAI_API_KEY:
        OPENAI_API_KEY=your-api-key-here
    - Change the MODEL_NAME if required

5. **Run the Streamlit app**
    streamlit run app.py

## Future Improvements
    - Better Table Parsing – Extend load_pdf() to handle different table formats for more accurate data extraction.

    - Persistent Multi-Document Memory – Allow users to retain memory across multiple documents and cross-query them.

    - Graphical Insights & Comparisons – Add visualization support to compare data across documents.

