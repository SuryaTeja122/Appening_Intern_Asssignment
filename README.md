# Appening AI Intern Assignment: LangGraph & RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot using **LangGraph**, **ChromaDB** (as a local alternative to Pinecone to save API costs), and **FastAPI**. It answers queries strictly based on the provided "Agentic AI eBook" PDF.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Set up Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory (you can rename `.env.example`) and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Provide the eBook**
   Ensure the file `Ebook-Agentic-AI.pdf` is in the root directory of the project.

6. **Ingest the PDF Document**
   Run the ingestion script to parse the PDF, generate embeddings, and store them in ChromaDB.
   ```bash
   python ingest.py
   ```

7. **Start the API Server**
   ```bash
   uvicorn app:app --reload
   ```

## Architecture Breakdown

- **Data Ingestion (`ingest.py`)**: Uses PyPDFLoader to load the PDF and RecursiveCharacterTextSplitter to split it into chunks of 1000 characters with 100 overlap. Embeddings are created via OpenAI and stored in a local Chroma vector database.
- **Orchestration Graph (`graph.py`)**: Built with LangGraph. It contains:
  - **Retrieve Node**: Queries the Chroma vector index for the top-3 most similar chunks.
  - **Generate Node**: Uses an LLM (`gpt-4o-mini`) to synthesize a grounded answer based *only* on the retrieved context. It also computes a confidence score to prevent hallucinations.
- **API Interface (`app.py`)**: Exposes a FastAPI endpoint (`/query`) that triggers the LangGraph workflow and returns a JSON payload with the query, final answer, retrieved context chunks, and confidence score.

## Testing the API
You can test the API using cURL or via the FastAPI interactive docs at `http://127.0.0.1:8000/docs`.

Example Request:
```bash
curl -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d "{\"query\": \"What is Agentic AI?\"}"
```
