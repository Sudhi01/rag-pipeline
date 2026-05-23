# RAG Pipeline with Hybrid Search

A production-grade Retrieval-Augmented Generation system that ingests technical documentation, indexes it with both dense vector and sparse keyword search, retrieves the most relevant context for any question, and generates grounded answers with inline source citations verified by an LLM judge.

## Eval Results

| Metric | Score |
|--------|-------|
| Answer Correctness | 83.5% |
| Source Retrieval | 75.0% |
| Avg Confidence | 89.3% |
| Citation Accuracy | 100.0% |

## Chunking Strategy Comparison

| Strategy | Chunks | Correctness | Source Retrieval | Citation Accuracy |
|----------|--------|-------------|------------------|-------------------|
| Fixed | 3,771 | 89.0% | 68.8% | 100.0% |
| Recursive | 4,284 | 87.1% | 81.2% | 100.0% |
| Semantic | 1,465 | 85.1% | 75.0% | 100.0% |

Recursive chunking was selected because it achieves the best source retrieval (81.2%) by respecting markdown heading boundaries — critical for technical documentation where related content lives under the same heading.

## Architecture
Documents (MD/PDF/TXT)
↓
Document Loader → 3 Chunking Strategies
↓
OpenAI Embeddings → ChromaDB (Dense)
↓              BM25 Index (Sparse)
↓
Hybrid Retrieval (RRF Fusion)
↓
LLM Reranker (top 20 → top 5)
↓
Grounded Generation (GPT-4o)
↓
Citation Verification (LLM-as-Judge)
↓
Confidence Scoring + Response

## What Makes This Different From a Basic RAG Demo

- **Hybrid search** — combines dense vector search with BM25 sparse search via Reciprocal Rank Fusion. BM25 catches exact keyword matches (function names, config keys, error codes) that semantic search misses.
- **Citation verification** — every citation is verified by an LLM judge after generation. Unsupported citations are flagged. Most RAG systems skip this entirely.
- **Confident "I don't know"** — when retrieval confidence is below threshold, the system refuses to answer rather than hallucinating. Composite confidence score of 11.6% on out-of-domain queries.
- **Three chunking strategies** with eval-driven selection — fixed, recursive, and semantic, benchmarked against a 30-question golden dataset.
- **Full eval framework** — automated metrics for correctness, source retrieval, faithfulness, and citation accuracy.

## Tech Stack

| Component | Tool |
|-----------|------|
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB |
| Sparse Search | BM25 via rank-bm25 |
| LLM | GPT-4o |
| Chunking | LangChain text splitters |
| API | FastAPI |
| Dashboard | Streamlit |
| Containerization | Docker |

## Quick Start

### Option 1 — Local

```bash
git clone <your-repo>
cd rag-pipeline
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Add your OpenAI API key
echo OPENAI_API_KEY=your_key_here > .env

# Seed the index
python seed.py

# Start the API
uvicorn main:app --reload

# Start the dashboard (new terminal)
python -m streamlit run dashboard.py
```

### Option 2 — Docker

```bash
git clone <your-repo>
cd rag-pipeline
echo OPENAI_API_KEY=your_key_here > .env
docker-compose up --build
python seed.py
```

Open:
- API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /v1/ask | Ask a question, get answer with citations |
| GET | /v1/documents | List indexed documents and chunk count |
| POST | /v1/ingest | Ingest new documents |
| GET | /v1/eval/results | Get evaluation results |

## Example Response

```json
{
  "question": "How do I add CORS support to FastAPI?",
  "answer": "To add CORS support, use CORSMiddleware [1]. Import it from fastapi.middleware.cors and add it to your app [2].",
  "confidence": {
    "retrieval_confidence": 0.84,
    "citation_coverage": 1.0,
    "completeness": 0.9,
    "composite_score": 0.916
  },
  "citations": {
    "1": {
      "source": "cors.md",
      "supported": true,
      "confidence": 1.0
    }
  }
}
```

## Project Structure
rag-pipeline/
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py    # Multi-format document loading
│   │   ├── chunker.py            # 3 chunking strategies
│   │   ├── embedder.py           # OpenAI embeddings + ChromaDB
│   │   └── bm25_index.py         # Sparse keyword index
│   ├── retrieval/
│   │   └── retriever.py          # Hybrid search + RRF + reranker
│   ├── generation/
│   │   └── generator.py          # Grounded generation + citation verification
│   └── evaluation/
│       ├── evaluator.py          # Golden dataset + automated metrics
│       └── chunking_comparison.py # Strategy benchmarking
├── main.py                       # FastAPI service
├── dashboard.py                  # Streamlit dashboard
├── seed.py                       # Index seeding script
├── Dockerfile
├── docker-compose.yml
└── README.md

## Key Design Decisions

**Why hybrid search over dense-only?**
Dense search alone missed exact technical terms like function names and config keys. BM25 sparse search catches these precisely. RRF fusion combines both ranked lists without requiring score normalization.

**Why recursive chunking?**
Benchmarked all three strategies. Recursive achieved 81.2% source retrieval vs 68.8% for fixed because it preserves heading structure — keeping related content together improves retrieval precision.

**Why citation verification?**
LLMs cite sources that don't actually support their claims. Verifying every citation with an LLM judge before returning the response catches unsupported claims. This is the quality layer most RAG systems skip.

**Why a confidence score?**
A single answer without uncertainty quantification is dangerous in production. The composite score (retrieval + citation coverage + completeness) gives users a signal about when to trust the answer vs. verify manually.

