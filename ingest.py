import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def ingest_pdf(pdf_path: str, persist_directory: str = "./chroma_db"):
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found. Please place the eBook PDF in the project directory.")
        return
        
    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(docs)
    
    print("Generating embeddings and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("Ingestion complete. Vector database created at", persist_directory)

if __name__ == "__main__":
    ingest_pdf("Ebook-Agentic-AI.pdf")
