import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from src.ingestion.bm25_index import bm25_search

load_dotenv()
client = OpenAI()

CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "rag_documents"

def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def embed_query(query: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    return response.data[0].embedding

def dense_search(query: str, top_k: int = 10) -> list[dict]:
    collection = get_collection()
    embedding = embed_query(query)
    
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    chunks = []
    for i in range(len(results["ids"][0])):
        similarity = 1 - results["distances"][0][i]
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "dense_score": similarity,
            "rank": i + 1
        })
    return chunks

def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    k: int = 60
) -> list[dict]:
    scores = {}
    all_chunks = {}

    for rank, chunk in enumerate(dense_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + dense_weight * (1 / (k + rank + 1))
        all_chunks[cid] = chunk

    for rank, chunk in enumerate(sparse_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + sparse_weight * (1 / (k + rank + 1))
        if cid not in all_chunks:
            all_chunks[cid] = chunk

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused = []
    for rank, cid in enumerate(sorted_ids):
        chunk = all_chunks[cid].copy()
        chunk["rrf_score"] = scores[cid]
        chunk["rank"] = rank + 1
        fused.append(chunk)

    return fused

def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    if not chunks:
        return []

    candidates = chunks[:20]
    
    scored = []
    for chunk in candidates:
        prompt = f"""Rate how relevant this text is for answering the query.

Query: {query}

Text: {chunk['content'][:500]}

Return only a number from 0.0 to 1.0 where:
1.0 = perfectly relevant
0.0 = completely irrelevant

Number:"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            score = max(0.0, min(1.0, score))
        except ValueError:
            score = chunk.get("rrf_score", 0.0)
        
        scored.append({**chunk, "rerank_score": score})

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]

def hybrid_search(
    query: str,
    top_k: int = 5,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    use_reranker: bool = True
) -> dict:
    print(f"\nSearching: '{query}'")
    
    # Dense retrieval
    dense_results = dense_search(query, top_k=10)
    print(f"  Dense results: {len(dense_results)}")
    
    # Sparse retrieval
    sparse_results = bm25_search(query, top_k=10)
    print(f"  Sparse results: {len(sparse_results)}")
    
    # Fusion
    fused = reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight
    )
    print(f"  After fusion: {len(fused)}")
    
    # Reranking
    if use_reranker:
        final = rerank(query, fused, top_k=top_k)
        print(f"  After reranking: {len(final)}")
    else:
        final = fused[:top_k]
    
    return {
        "query": query,
        "results": final,
        "dense_count": len(dense_results),
        "sparse_count": len(sparse_results),
        "fused_count": len(fused)
    }


if __name__ == "__main__":
    query = "how to add authentication to FastAPI"
    
    print("=== HYBRID SEARCH TEST ===")
    result = hybrid_search(query, top_k=5, use_reranker=True)
    
    print(f"\nTop {len(result['results'])} results after reranking:\n")
    for r in result["results"]:
        print(f"  Rank {r['rank']} | Rerank: {r.get('rerank_score', 0):.3f} | RRF: {r.get('rrf_score', 0):.4f}")
        print(f"  File: {r['metadata'].get('filename', 'unknown')}")
        print(f"  Preview: {r['content'][:200]}")
        print()
    
    print("\n=== DENSE ONLY vs HYBRID COMPARISON ===")
    dense_only = dense_search(query, top_k=5)
    print(f"\nDense only top 5:")
    for r in dense_only:
        print(f"  Rank {r['rank']} | Score: {r['dense_score']:.3f} | File: {r['metadata'].get('filename', 'unknown')}")
    
    print(f"\nHybrid top 5:")
    for r in result["results"]:
        print(f"  Rank {r['rank']} | Score: {r.get('rerank_score', 0):.3f} | File: {r['metadata'].get('filename', 'unknown')}")