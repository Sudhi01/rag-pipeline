import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def run_chunking_comparison():
    from src.ingestion.document_loader import load_documents_from_directory
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.embedder import embed_and_store_chunks
    from src.ingestion.bm25_index import build_bm25_index
    from src.evaluation.evaluator import (
        score_answer_correctness,
        check_source_retrieval,
        GOLDEN_DATASET
    )
    from src.generation.generator import ask
    import chromadb
    import shutil

    strategies = ["fixed", "recursive", "semantic"]
    test_questions = GOLDEN_DATASET[:8]
    results = {}

    docs = load_documents_from_directory("fastapi/docs/en/docs")

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing strategy: {strategy.upper()}")
        print('='*60)

        # Reset chroma for clean comparison
        chroma_path = f"data/chroma_{strategy}"
        if Path(chroma_path).exists():
            shutil.rmtree(chroma_path)

        # Build index for this strategy
        chunks = chunk_documents(docs, strategy=strategy)

        # Store in strategy-specific collection
        import chromadb as cb
        chroma_client = cb.PersistentClient(path=chroma_path)
        collection = chroma_client.get_or_create_collection(
            name=f"rag_{strategy}",
            metadata={"hnsw:space": "cosine"}
        )

        from openai import OpenAI
        oai = OpenAI()

        print(f"Embedding {len(chunks)} chunks...")
        texts = [c.content for c in chunks]
        all_embeddings = []
        for i in range(0, len(texts), 100):
            batch = texts[i:i+100]
            response = oai.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            all_embeddings.extend([r.embedding for r in response.data])
            print(f"  Batch {i//100+1}/{(len(texts)-1)//100+1}")

        for i, (chunk, emb) in enumerate(zip(chunks, all_embeddings)):
            meta = {k: str(v) for k, v in chunk.metadata.items()}
            meta["strategy"] = chunk.strategy
            collection.add(
                ids=[chunk.chunk_id],
                embeddings=[emb],
                documents=[chunk.content],
                metadatas=[meta]
            )

        build_bm25_index(chunks)

        # Run eval questions
        strategy_results = []
        correctness_scores = []
        source_scores = []
        confidence_scores = []
        citation_scores = []

        for i, item in enumerate(test_questions):
            print(f"  [{i+1}/{len(test_questions)}] {item['question'][:50]}...")

            # Use strategy-specific chroma
            from src.retrieval import retriever as ret
            original_path = ret.CHROMA_PATH
            ret.CHROMA_PATH = chroma_path

            original_collection = ret.COLLECTION_NAME
            ret.COLLECTION_NAME = f"rag_{strategy}"

            try:
                result = ask(item["question"])

                if item.get("should_not_answer"):
                    correctness = 1.0 if "don't have enough information" in result["answer"].lower() else 0.0
                else:
                    scores = score_answer_correctness(
                        item["question"],
                        result["answer"],
                        item["expected_topics"]
                    )
                    correctness = scores["overall"]

                source_score = check_source_retrieval(
                    result["sources"],
                    item["expected_sources"]
                )

                total_cit = len(result["citations"])
                supported_cit = sum(
                    1 for v in result["citations"].values()
                    if v.get("supported", False)
                )
                cit_acc = supported_cit / total_cit if total_cit > 0 else 0

                correctness_scores.append(correctness)
                source_scores.append(source_score)
                confidence_scores.append(result["confidence"]["composite_score"])
                citation_scores.append(cit_acc)

                strategy_results.append({
                    "question": item["question"],
                    "correctness": correctness,
                    "source_score": source_score,
                    "confidence": result["confidence"]["composite_score"],
                    "citation_accuracy": cit_acc
                })

            finally:
                ret.CHROMA_PATH = original_path
                ret.COLLECTION_NAME = original_collection

        results[strategy] = {
            "chunks_count": len(chunks),
            "avg_correctness": round(sum(correctness_scores)/len(correctness_scores), 3),
            "avg_source_retrieval": round(sum(source_scores)/len(source_scores), 3),
            "avg_confidence": round(sum(confidence_scores)/len(confidence_scores), 3),
            "avg_citation_accuracy": round(sum(citation_scores)/len(citation_scores), 3),
            "per_question": strategy_results
        }

    # Print comparison table
    print(f"\n{'='*60}")
    print("CHUNKING STRATEGY COMPARISON")
    print('='*60)
    print(f"{'Metric':<25} {'Fixed':>10} {'Recursive':>12} {'Semantic':>10}")
    print('-'*60)
    print(f"{'Chunks generated':<25} {results['fixed']['chunks_count']:>10} {results['recursive']['chunks_count']:>12} {results['semantic']['chunks_count']:>10}")
    print(f"{'Avg correctness':<25} {results['fixed']['avg_correctness']:>10.1%} {results['recursive']['avg_correctness']:>12.1%} {results['semantic']['avg_correctness']:>10.1%}")
    print(f"{'Avg source retrieval':<25} {results['fixed']['avg_source_retrieval']:>10.1%} {results['recursive']['avg_source_retrieval']:>12.1%} {results['semantic']['avg_source_retrieval']:>10.1%}")
    print(f"{'Avg confidence':<25} {results['fixed']['avg_confidence']:>10.1%} {results['recursive']['avg_confidence']:>12.1%} {results['semantic']['avg_confidence']:>10.1%}")
    print(f"{'Citation accuracy':<25} {results['fixed']['avg_citation_accuracy']:>10.1%} {results['recursive']['avg_citation_accuracy']:>12.1%} {results['semantic']['avg_citation_accuracy']:>10.1%}")

    # Save results
    output_path = "data/chunking_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_path}")

    return results


if __name__ == "__main__":
    run_chunking_comparison()