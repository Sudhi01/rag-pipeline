import json
import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from src.ingestion.chunker import Chunk

load_dotenv()

BM25_PATH = "data/bm25_index.pkl"
CHUNKS_PATH = "data/chunks_store.json"

def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s_]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]

def build_bm25_index(chunks: list[Chunk]) -> BM25Okapi:
    print(f"Building BM25 index for {len(chunks)} chunks...")
    
    tokenized_corpus = [tokenize(chunk.content) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save index
    Path("data").mkdir(exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    
    # Save chunks separately for retrieval
    chunks_data = [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "content": chunk.content,
            "strategy": chunk.strategy,
            "chunk_index": chunk.chunk_index,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False)
    
    print(f"BM25 index saved to {BM25_PATH}")
    print(f"Chunks store saved to {CHUNKS_PATH}")
    return bm25

def load_bm25_index():
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    return bm25, chunks_data

def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    bm25, chunks_data = load_bm25_index()
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                **chunks_data[idx],
                "bm25_score": float(scores[idx]),
                "rank": len(results) + 1
            })
    
    return results


if __name__ == "__main__":
    from src.ingestion.document_loader import load_documents_from_directory
    from src.ingestion.chunker import chunk_documents

    docs = load_documents_from_directory("fastapi/docs/en/docs")
    chunks = chunk_documents(docs, strategy="recursive")
    
    bm25 = build_bm25_index(chunks)
    
    # Test search
    print("\n--- Testing BM25 Search ---")
    query = "how to add authentication to FastAPI"
    results = bm25_search(query, top_k=5)
    
    print(f"\nQuery: '{query}'")
    print(f"Top {len(results)} results:\n")
    for r in results:
        print(f"  Rank {r['rank']} | Score: {r['bm25_score']:.4f}")
        print(f"  File: {r['metadata'].get('filename', 'unknown')}")
        print(f"  Preview: {r['content'][:150]}")
        print()