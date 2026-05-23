import os
import json
import hashlib
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from src.ingestion.chunker import Chunk

load_dotenv()
client = OpenAI()

CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "rag_documents"

def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(embeddings)
        print(f"  Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
    return all_embeddings

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a ** 2 for a in vec1) ** 0.5
    norm2 = sum(b ** 2 for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def is_duplicate(embedding: list[float], collection, threshold: float = 0.95) -> bool:
    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["distances"]
        )
        if results["distances"] and results["distances"][0]:
            distance = results["distances"][0][0]
            similarity = 1 - distance
            return similarity > threshold
    except Exception:
        pass
    return False

def embed_and_store_chunks(chunks: list[Chunk], skip_duplicates: bool = True) -> dict:
    collection = get_chroma_collection()
    
    existing_count = collection.count()
    print(f"Existing chunks in store: {existing_count}")
    print(f"New chunks to process: {len(chunks)}")

    texts = [chunk.content for chunk in chunks]
    print(f"\nGenerating embeddings...")
    embeddings = embed_texts(texts)

    added = 0
    skipped = 0

    print(f"\nStoring chunks (duplicate check: {skip_duplicates})...")
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        if skip_duplicates and existing_count > 0:
            if is_duplicate(embedding, collection):
                skipped += 1
                continue

        metadata = {k: str(v) for k, v in chunk.metadata.items()}
        metadata["strategy"] = chunk.strategy
        metadata["chunk_index"] = str(chunk.chunk_index)

        collection.add(
            ids=[chunk.chunk_id],
            embeddings=[embedding],
            documents=[chunk.content],
            metadatas=[metadata]
        )
        added += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(chunks)} chunks...")

    print(f"\nDone. Added: {added} | Skipped (duplicates): {skipped}")
    print(f"Total in store: {collection.count()}")
    
    return {
        "added": added,
        "skipped": skipped,
        "total": collection.count()
    }


if __name__ == "__main__":
    from src.ingestion.document_loader import load_documents_from_directory
    from src.ingestion.chunker import chunk_documents

    docs = load_documents_from_directory("fastapi/docs/en/docs")
    chunks = chunk_documents(docs, strategy="recursive")
    
    print(f"\nEmbedding {len(chunks)} chunks...")
    result = embed_and_store_chunks(chunks)
    
    print(f"\nFinal result: {result}")