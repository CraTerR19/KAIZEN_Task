# Kaizen-Task-RAG

My Task is to build a chatbot that reads all the files from a folder on my local machine, indexes their content into a vector store, and lets me ask questions about them in a terminal chat loop. When I ask a question, the system should retrieve the most relevant chunks from my files and use them as context for the LLM's answer, so the answer is always grounded in my actual files, not made up.


## Working

1. My `Sample_data.txt` file is in the `docs/` folder
2. `ingest.py` reads it, splits into chunks, and stores vector embeddings in ChromaDB
3. `main.py` lets me ask questions - it fetches relevant chunks and sends them to Gemini for final response

## Setup

Install dependencies:
pip install below dependencies
langchain
langchain-community 
langchain-google-genai 
chromadb
sentence-transformers 
python-dotenv
langchain-classic

pip install -r requirements.txt

Create a `.env` file with your Gemini API key (get it from Google AI Studio):

GEMINI_API_KEY

## Running

First ingest my documents:

python ingest.py

Then start chatting:

python main.py

Type `exit` to quit.

## Files

- `ingest.py`- Reads docs, creates embeddings, saves to ChromaDB
- `main.py` - Interactive Q&A loop using the vector store and Gemini
- `docs/` - Contains Sample_data which is the overview of th project
- `chroma_db/` - Auto-generated vector database

Technologies:

1. Python
2. LangChain
3. ChromaDB
4. Any LLM API Gemini for local inference