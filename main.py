from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="RAG Pipeline API",
    description="Production RAG system with hybrid search over FastAPI documentation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    use_reranker: Optional[bool] = True

class IngestRequest(BaseModel):
    directory: str
    strategy: Optional[str] = "recursive"

@app.get("/")
def root():
    return {
        "name": "RAG Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "POST /v1/ask",
            "GET /v1/documents",
            "POST /v1/ingest",
            "GET /v1/eval/results"
        ]
    }

@app.get("/debug")
def debug():
    key = os.environ.get("OPENAI_API_KEY", "NOT FOUND")
    return {
        "key_found": key != "NOT FOUND",
        "key_prefix": key[:10] if key != "NOT FOUND" else "NOT FOUND",
        "env_vars": list(os.environ.keys())
    }

@app.post("/v1/ask")
def ask_question(request: QuestionRequest):
    try:
        from src.generation.generator import ask
        result = ask(
            query=request.question,
            top_k=request.top_k
        )
        return {
            "question": result["query"],
            "answer": result["answer"],
            "confidence": result["confidence"],
            "citations": result["citations"],
            "sources": result["sources"],
            "chunks_used": result["chunks_used"]
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Index not built yet. Call POST /v1/ingest first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/documents")
def list_documents():
    try:
        import chromadb
        chroma_client = chromadb.PersistentClient(path="data/chroma")
        collection = chroma_client.get_or_create_collection("rag_documents")
        count = collection.count()

        chunks_path = Path("data/chunks_store.json")
        sources = []
        if chunks_path.exists():
            with open(chunks_path) as f:
                chunks = json.load(f)
            sources = list(set(
                c["metadata"].get("filename", "unknown")
                for c in chunks
            ))

        return {
            "total_chunks": count,
            "total_documents": len(sources),
            "documents": sorted(sources)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ingest")
def ingest_documents(request: IngestRequest):
    try:
        from src.ingestion.document_loader import load_documents_from_directory
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.embedder import embed_and_store_chunks
        from src.ingestion.bm25_index import build_bm25_index

        docs = load_documents_from_directory(request.directory)
        if not docs:
            raise HTTPException(status_code=400, detail="No documents found")

        chunks = chunk_documents(docs, strategy=request.strategy)
        embed_result = embed_and_store_chunks(chunks)
        build_bm25_index(chunks)

        return {
            "status": "success",
            "documents_loaded": len(docs),
            "chunks_created": len(chunks),
            "chunks_added": embed_result["added"],
            "chunks_skipped": embed_result["skipped"],
            "strategy": request.strategy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/eval/results")
def get_eval_results():
    results_path = Path("data/eval_results.json")
    if not results_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No eval results found. Run evaluator first."
        )
    with open(results_path) as f:
        return json.load(f)