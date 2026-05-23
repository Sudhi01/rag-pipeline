import re
from openai import OpenAI
from dotenv import load_dotenv
from src.retrieval.retriever import hybrid_search

load_dotenv()
client = OpenAI()

def build_context(chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks):
        source = chunk["metadata"].get("filename", "unknown")
        context_parts.append(f"[{i+1}] Source: {source}\n{chunk['content']}")
    return "\n\n---\n\n".join(context_parts)

def generate_answer(query: str, chunks: list[dict]) -> str:
    context = build_context(chunks)
    
    system_prompt = """You are a helpful assistant that answers questions about FastAPI documentation.

STRICT RULES:
1. Answer ONLY using the provided context below
2. Cite sources using bracketed numbers like [1], [2] after each claim
3. If the context doesn't contain enough information, say exactly: "I don't have enough information in the provided context to answer this question fully."
4. Never make up information not present in the context
5. Be concise and precise"""

    user_prompt = f"""Context:
{context}

Question: {query}

Answer (with citations):"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def extract_citations(answer: str) -> list[int]:
    pattern = r'\[(\d+)\]'
    matches = re.findall(pattern, answer)
    return list(set(int(m) for m in matches))

def verify_citation(claim_context: str, chunk_content: str) -> dict:
    prompt = f"""Does this source text support the claim?

Source text: {chunk_content[:600]}

Claim context: {claim_context}

Answer with JSON only:
{{"supported": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150
    )
    
    text = response.choices[0].message.content.strip()
    
    try:
        import json
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return {"supported": True, "confidence": 0.5, "reason": "Could not parse"}

def verify_all_citations(answer: str, chunks: list[dict]) -> dict:
    cited_indices = extract_citations(answer)
    results = {}
    
    for idx in cited_indices:
        chunk_idx = idx - 1
        if 0 <= chunk_idx < len(chunks):
            verification = verify_citation(answer, chunks[chunk_idx]["content"])
            results[idx] = {
                "chunk_id": chunks[chunk_idx]["chunk_id"],
                "source": chunks[chunk_idx]["metadata"].get("filename", "unknown"),
                **verification
            }
        else:
            results[idx] = {
                "supported": False,
                "confidence": 0.0,
                "reason": "Citation index out of range"
            }
    
    return results

def compute_confidence_score(
    chunks: list[dict],
    citation_results: dict,
    answer: str
) -> dict:
    # Retrieval confidence
    if chunks:
        avg_rerank = sum(
            c.get("rerank_score", c.get("rrf_score", 0))
            for c in chunks
        ) / len(chunks)
        retrieval_confidence = min(avg_rerank, 1.0)
    else:
        retrieval_confidence = 0.0

    # Citation coverage
    if citation_results:
        supported = sum(
            1 for v in citation_results.values() if v.get("supported", False)
        )
        citation_coverage = supported / len(citation_results)
    else:
        citation_coverage = 0.0

    # Answer completeness
    if "don't have enough information" in answer.lower():
        completeness = 0.3
    elif len(answer) > 200:
        completeness = 0.9
    else:
        completeness = 0.7

    composite = (
        retrieval_confidence * 0.4 +
        citation_coverage * 0.4 +
        completeness * 0.2
    )

    return {
        "retrieval_confidence": round(retrieval_confidence, 3),
        "citation_coverage": round(citation_coverage, 3),
        "completeness": round(completeness, 3),
        "composite_score": round(composite, 3)
    }

def handle_low_confidence(query: str, chunks: list[dict]) -> str:
    sources = list(set(
        c["metadata"].get("filename", "unknown") for c in chunks
    ))
    return (
        f"I couldn't find a confident answer to: '{query}'\n\n"
        f"Related documents that might help:\n" +
        "\n".join(f"  - {s}" for s in sources[:5]) +
        "\n\nConsider checking these files manually."
    )

def ask(query: str, top_k: int = 5) -> dict:
    # Retrieve
    search_result = hybrid_search(query, top_k=top_k, use_reranker=True)
    chunks = search_result["results"]
    
    if not chunks:
        return {
            "query": query,
            "answer": "No relevant documents found.",
            "citations": {},
            "confidence": {"composite_score": 0.0},
            "chunks_used": 0
        }
    
    # Generate
    print("  Generating answer...")
    answer = generate_answer(query, chunks)
    
    # Verify citations
    print("  Verifying citations...")
    citation_results = verify_all_citations(answer, chunks)
    
    # Score confidence
    confidence = compute_confidence_score(chunks, citation_results, answer)
    
    # Handle low confidence
    if confidence["composite_score"] < 0.3:
        answer = handle_low_confidence(query, chunks)
    
    return {
        "query": query,
        "answer": answer,
        "citations": citation_results,
        "confidence": confidence,
        "chunks_used": len(chunks),
        "sources": list(set(
            c["metadata"].get("filename", "unknown") for c in chunks
        ))
    }


if __name__ == "__main__":
    queries = [
        "How do I add OAuth2 authentication to FastAPI?",
        "What is dependency injection in FastAPI?",
        "How do I deploy FastAPI with Docker?",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)
        
        result = ask(query)
        
        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nCONFIDENCE: {result['confidence']}")
        print(f"\nCITATIONS VERIFIED: {len(result['citations'])}")
        for idx, citation in result['citations'].items():
            status = "✅" if citation.get('supported') else "❌"
            print(f"  [{idx}] {status} {citation.get('source')} (confidence: {citation.get('confidence', 0):.2f})")
        print(f"\nSOURCES USED: {result['sources']}")