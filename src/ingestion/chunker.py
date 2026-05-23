import re
from dataclasses import dataclass, field
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.document_loader import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    content: str
    strategy: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

def fixed_size_chunking(doc: Document, chunk_size: int = 512, overlap: int = 50) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = splitter.split_text(doc.content)
    chunks = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_fixed_{i}",
            doc_id=doc.doc_id,
            content=text.strip(),
            strategy="fixed",
            chunk_index=i,
            metadata={
                **doc.metadata,
                "chunk_size": len(text),
                "strategy": "fixed"
            }
        ))
    return chunks

def recursive_chunking(doc: Document, chunk_size: int = 512, overlap: int = 50) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
    )
    texts = splitter.split_text(doc.content)
    chunks = []
    current_heading = "Introduction"

    for i, text in enumerate(texts):
        if not text.strip():
            continue
        heading_match = re.search(r'^#{1,4}\s+(.+)', text, re.MULTILINE)
        if heading_match:
            current_heading = heading_match.group(1).strip()
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_recursive_{i}",
            doc_id=doc.doc_id,
            content=text.strip(),
            strategy="recursive",
            chunk_index=i,
            metadata={
                **doc.metadata,
                "section_heading": current_heading,
                "chunk_size": len(text),
                "strategy": "recursive"
            }
        ))
    return chunks

def semantic_chunking(doc: Document, max_chunk_size: int = 1000) -> list[Chunk]:
    paragraphs = [p.strip() for p in doc.content.split("\n\n") if p.strip()]
    
    if not paragraphs:
        return fixed_size_chunking(doc)

    chunks = []
    current_chunk = ""
    chunk_index = 0

    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chunk_size:
            current_chunk += "\n\n" + para if current_chunk else para
        else:
            if current_chunk.strip():
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_semantic_{chunk_index}",
                    doc_id=doc.doc_id,
                    content=current_chunk.strip(),
                    strategy="semantic",
                    chunk_index=chunk_index,
                    metadata={
                        **doc.metadata,
                        "chunk_size": len(current_chunk),
                        "strategy": "semantic"
                    }
                ))
                chunk_index += 1
            current_chunk = para

    if current_chunk.strip():
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_semantic_{chunk_index}",
            doc_id=doc.doc_id,
            content=current_chunk.strip(),
            strategy="semantic",
            chunk_index=chunk_index,
            metadata={
                **doc.metadata,
                "chunk_size": len(current_chunk),
                "strategy": "semantic"
            }
        ))

    return chunks

def chunk_documents(docs: list[Document], strategy: str = "recursive") -> list[Chunk]:
    all_chunks = []
    strategy_map = {
        "fixed": fixed_size_chunking,
        "recursive": recursive_chunking,
        "semantic": semantic_chunking
    }
    
    chunker = strategy_map.get(strategy, recursive_chunking)
    
    for doc in docs:
        chunks = chunker(doc)
        all_chunks.extend(chunks)
    
    print(f"Strategy: {strategy} | Documents: {len(docs)} | Chunks: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    from src.ingestion.document_loader import load_documents_from_directory
    
    docs = load_documents_from_directory("fastapi/docs/en/docs")
    
    print("\n--- Testing all 3 chunking strategies ---\n")
    
    fixed_chunks = chunk_documents(docs, strategy="fixed")
    recursive_chunks = chunk_documents(docs, strategy="recursive")
    semantic_chunks = chunk_documents(docs, strategy="semantic")
    
    print(f"\nFirst recursive chunk preview:")
    print(f"  ID: {recursive_chunks[0].chunk_id}")
    print(f"  Strategy: {recursive_chunks[0].strategy}")
    print(f"  Content length: {len(recursive_chunks[0].content)}")
    print(f"  Preview: {recursive_chunks[0].content[:200]}")