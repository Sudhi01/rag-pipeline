"""
Run this once after deployment to index the documents.
Call POST /v1/ingest endpoint or run this script directly.
"""
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

def seed():
    # Clone FastAPI docs if not present
    if not os.path.exists("fastapi"):
        print("Cloning FastAPI docs...")
        subprocess.run([
            "git", "clone",
            "https://github.com/tiangolo/fastapi.git",
            "--depth=1"
        ], check=True)
        print("Cloned successfully")

    from src.ingestion.document_loader import load_documents_from_directory
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.embedder import embed_and_store_chunks
    from src.ingestion.bm25_index import build_bm25_index

    print("Loading documents...")
    docs = load_documents_from_directory("fastapi/docs/en/docs")
    
    print("Chunking...")
    chunks = chunk_documents(docs, strategy="recursive")
    
    print("Embedding and storing...")
    embed_and_store_chunks(chunks)
    
    print("Building BM25 index...")
    build_bm25_index(chunks)
    
    print("Done. RAG pipeline ready.")

if __name__ == "__main__":
    seed()

if __name__ == "__main__":
    seed()