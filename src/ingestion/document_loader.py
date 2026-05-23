import os
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import fitz  # pymupdf
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Document:
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)

def generate_doc_id(filepath: str) -> str:
    return hashlib.md5(filepath.encode()).hexdigest()[:12]

def load_markdown(filepath: Path) -> Optional[Document]:
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        if not content.strip():
            return None
        return Document(
            doc_id=generate_doc_id(str(filepath)),
            content=content,
            metadata={
                "source": str(filepath),
                "filename": filepath.name,
                "format": "markdown",
                "section": filepath.stem
            }
        )
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_pdf(filepath: Path) -> Optional[Document]:
    try:
        doc = fitz.open(str(filepath))
        content = ""
        for page in doc:
            content += page.get_text()
        doc.close()
        if not content.strip():
            return None
        return Document(
            doc_id=generate_doc_id(str(filepath)),
            content=content,
            metadata={
                "source": str(filepath),
                "filename": filepath.name,
                "format": "pdf",
                "section": filepath.stem
            }
        )
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_text(filepath: Path) -> Optional[Document]:
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        if not content.strip():
            return None
        return Document(
            doc_id=generate_doc_id(str(filepath)),
            content=content,
            metadata={
                "source": str(filepath),
                "filename": filepath.name,
                "format": "text",
                "section": filepath.stem
            }
        )
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_documents_from_directory(directory: str) -> list[Document]:
    docs = []
    path = Path(directory)
    
    if not path.exists():
        print(f"Directory not found: {directory}")
        return docs

    loaders = {
        ".md": load_markdown,
        ".txt": load_text,
        ".pdf": load_pdf,
        ".html": load_text,
    }

    files = list(path.rglob("*"))
    print(f"Scanning {len(files)} files in {directory}...")

    for filepath in files:
        if filepath.is_file() and filepath.suffix.lower() in loaders:
            doc = loaders[filepath.suffix.lower()](filepath)
            if doc:
                docs.append(doc)

    print(f"Loaded {len(docs)} documents successfully")
    return docs


if __name__ == "__main__":
    # Test the loader
    docs = load_documents_from_directory("fastapi/docs/en/docs")
    print(f"\nTotal documents loaded: {len(docs)}")
    if docs:
        print(f"\nFirst document preview:")
        print(f"  ID: {docs[0].doc_id}")
        print(f"  File: {docs[0].metadata['filename']}")
        print(f"  Format: {docs[0].metadata['format']}")
        print(f"  Content length: {len(docs[0].content)} chars")
        print(f"  Preview: {docs[0].content[:200]}")