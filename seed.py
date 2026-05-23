"""
Run this once after docker-compose up to index the documents.
python seed.py
"""
from src.ingestion.document_loader import load_documents_from_directory
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_and_store_chunks
from src.ingestion.bm25_index import build_bm25_index
from dotenv import load_dotenv

load_dotenv()

print("Seeding RAG pipeline with FastAPI documentation...")
docs = load_documents_from_directory("fastapi/docs/en/docs")
chunks = chunk_documents(docs, strategy="recursive")
embed_and_store_chunks(chunks)
build_bm25_index(chunks)
print("Done. RAG pipeline ready.")