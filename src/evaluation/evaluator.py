import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

GOLDEN_DATASET_PATH = "data/golden_dataset.json"

GOLDEN_DATASET = [
    {
        "id": "q001",
        "question": "How do I create a basic FastAPI application?",
        "expected_topics": ["app = FastAPI()", "uvicorn", "main.py"],
        "expected_sources": ["first-steps.md"],
        "difficulty": "easy"
    },
    {
        "id": "q002",
        "question": "How do I define path parameters in FastAPI?",
        "expected_topics": ["path parameter", "type annotation", "@app.get"],
        "expected_sources": ["path-params.md"],
        "difficulty": "easy"
    },
    {
        "id": "q003",
        "question": "What is the difference between path parameters and query parameters?",
        "expected_topics": ["path parameter", "query parameter", "default value"],
        "expected_sources": ["query-params.md", "path-params.md"],
        "difficulty": "medium"
    },
    {
        "id": "q004",
        "question": "How do I add request body validation in FastAPI?",
        "expected_topics": ["BaseModel", "Pydantic", "request body"],
        "expected_sources": ["body.md"],
        "difficulty": "medium"
    },
    {
        "id": "q005",
        "question": "How do I add OAuth2 authentication to FastAPI?",
        "expected_topics": ["OAuth2PasswordBearer", "token", "security"],
        "expected_sources": ["first-steps.md", "oauth2-jwt.md"],
        "difficulty": "hard"
    },
    {
        "id": "q006",
        "question": "What is dependency injection in FastAPI?",
        "expected_topics": ["Depends", "dependency", "inject"],
        "expected_sources": ["index.md", "dependencies.md"],
        "difficulty": "medium"
    },
    {
        "id": "q007",
        "question": "How do I deploy FastAPI with Docker?",
        "expected_topics": ["Dockerfile", "container", "image"],
        "expected_sources": ["docker.md"],
        "difficulty": "medium"
    },
    {
        "id": "q008",
        "question": "How do I add CORS support to FastAPI?",
        "expected_topics": ["CORSMiddleware", "origins", "middleware"],
        "expected_sources": ["cors.md"],
        "difficulty": "easy"
    },
    {
        "id": "q009",
        "question": "How do I handle file uploads in FastAPI?",
        "expected_topics": ["UploadFile", "File", "multipart"],
        "expected_sources": ["request-files.md"],
        "difficulty": "medium"
    },
    {
        "id": "q010",
        "question": "What is the fastest way to run FastAPI in production?",
        "expected_topics": ["uvicorn", "gunicorn", "workers"],
        "expected_sources": ["server-workers.md", "docker.md"],
        "difficulty": "hard"
    },
    {
        "id": "q011",
        "question": "How do I add background tasks in FastAPI?",
        "expected_topics": ["BackgroundTasks", "add_task"],
        "expected_sources": ["background-tasks.md"],
        "difficulty": "medium"
    },
    {
        "id": "q012",
        "question": "How do I use SQLAlchemy with FastAPI?",
        "expected_topics": ["SessionLocal", "Base", "database"],
        "expected_sources": ["sql-databases.md"],
        "difficulty": "hard"
    },
    {
        "id": "q013",
        "question": "How do I write tests for FastAPI applications?",
        "expected_topics": ["TestClient", "pytest", "test"],
        "expected_sources": ["testing.md"],
        "difficulty": "medium"
    },
    {
        "id": "q014",
        "question": "How do I add custom middleware to FastAPI?",
        "expected_topics": ["middleware", "BaseHTTPMiddleware", "request"],
        "expected_sources": ["middleware.md"],
        "difficulty": "medium"
    },
    {
        "id": "q015",
        "question": "What are response models in FastAPI?",
        "expected_topics": ["response_model", "Pydantic", "output"],
        "expected_sources": ["response-model.md"],
        "difficulty": "easy"
    },
    {
        "id": "q016",
        "question": "How do I handle errors and exceptions in FastAPI?",
        "expected_topics": ["HTTPException", "exception_handler", "status_code"],
        "expected_sources": ["handling-errors.md"],
        "difficulty": "medium"
    },
    {
        "id": "q017",
        "question": "How do I use WebSockets in FastAPI?",
        "expected_topics": ["WebSocket", "websocket", "connect"],
        "expected_sources": ["websockets.md"],
        "difficulty": "hard"
    },
    {
        "id": "q018",
        "question": "What are FastAPI routers and how do I use them?",
        "expected_topics": ["APIRouter", "include_router", "prefix"],
        "expected_sources": ["bigger-applications.md"],
        "difficulty": "medium"
    },
    {
        "id": "q019",
        "question": "How do I add rate limiting to FastAPI?",
        "expected_topics": ["middleware", "rate limit", "request"],
        "expected_sources": ["middleware.md"],
        "difficulty": "hard"
    },
    {
        "id": "q020",
        "question": "How do I return custom HTTP status codes in FastAPI?",
        "expected_topics": ["status_code", "Response", "HTTP"],
        "expected_sources": ["response-status-code.md"],
        "difficulty": "easy"
    },
    {
        "id": "q021",
        "question": "What is the purpose of Pydantic in FastAPI?",
        "expected_topics": ["validation", "BaseModel", "type hints"],
        "expected_sources": ["body.md", "features.md"],
        "difficulty": "easy"
    },
    {
        "id": "q022",
        "question": "How do I add API versioning in FastAPI?",
        "expected_topics": ["router", "prefix", "version"],
        "expected_sources": ["bigger-applications.md"],
        "difficulty": "hard"
    },
    {
        "id": "q023",
        "question": "How do I use async functions in FastAPI?",
        "expected_topics": ["async", "await", "async def"],
        "expected_sources": ["async.md"],
        "difficulty": "medium"
    },
    {
        "id": "q024",
        "question": "How do I add JWT token authentication in FastAPI?",
        "expected_topics": ["JWT", "jose", "token"],
        "expected_sources": ["oauth2-jwt.md"],
        "difficulty": "hard"
    },
    {
        "id": "q025",
        "question": "What is OpenAPI and how does FastAPI use it?",
        "expected_topics": ["OpenAPI", "Swagger", "docs"],
        "expected_sources": ["features.md", "index.md"],
        "difficulty": "easy"
    },
    {
        "id": "q026",
        "question": "How do I handle form data in FastAPI?",
        "expected_topics": ["Form", "form data", "OAuth2PasswordRequestForm"],
        "expected_sources": ["request-forms.md"],
        "difficulty": "medium"
    },
    {
        "id": "q027",
        "question": "How do I add static files support in FastAPI?",
        "expected_topics": ["StaticFiles", "mount", "static"],
        "expected_sources": ["static-files.md"],
        "difficulty": "easy"
    },
    {
        "id": "q028",
        "question": "How do I use environment variables in FastAPI?",
        "expected_topics": ["BaseSettings", "env", "config"],
        "expected_sources": ["settings.md"],
        "difficulty": "medium"
    },
    {
        "id": "q029",
        "question": "How do I add request validation with headers in FastAPI?",
        "expected_topics": ["Header", "header parameter", "validation"],
        "expected_sources": ["header-params.md"],
        "difficulty": "medium"
    },
    {
        "id": "q030",
        "question": "This question has no answer in the FastAPI documentation about quantum computing algorithms",
        "expected_topics": [],
        "expected_sources": [],
        "difficulty": "unanswerable",
        "should_not_answer": True
    }
]

def save_golden_dataset():
    Path("data").mkdir(exist_ok=True)
    with open(GOLDEN_DATASET_PATH, "w") as f:
        json.dump(GOLDEN_DATASET, f, indent=2)
    print(f"Saved {len(GOLDEN_DATASET)} golden questions to {GOLDEN_DATASET_PATH}")

def score_answer_correctness(question: str, answer: str, expected_topics: list[str]) -> dict:
    topics_str = ", ".join(expected_topics) if expected_topics else "general knowledge"
    
    prompt = f"""Evaluate this answer to a FastAPI documentation question.

Question: {question}
Expected topics to cover: {topics_str}
Answer given: {answer[:800]}

Rate on these dimensions (each 0.0 to 1.0):
1. correctness: Is the answer factually correct?
2. relevance: Does it actually answer the question?
3. topic_coverage: Does it cover the expected topics?

Return JSON only:
{{"correctness": 0.0, "relevance": 0.0, "topic_coverage": 0.0, "overall": 0.0}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100
    )
    
    try:
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return {"correctness": 0.5, "relevance": 0.5, "topic_coverage": 0.5, "overall": 0.5}

def check_source_retrieval(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    if not expected_sources:
        return 1.0
    hits = sum(1 for s in expected_sources if s in retrieved_sources)
    return hits / len(expected_sources)

def run_evaluation(sample_size: int = 10) -> dict:
    from src.generation.generator import ask
    
    save_golden_dataset()
    dataset = GOLDEN_DATASET[:sample_size]
    
    print(f"\nRunning evaluation on {len(dataset)} questions...")
    print("=" * 60)
    
    results = []
    
    for i, item in enumerate(dataset):
        print(f"\n[{i+1}/{len(dataset)}] {item['question'][:60]}...")
        
        result = ask(item["question"])
        
        # Score correctness
        if item.get("should_not_answer"):
            answered = "don't have enough information" not in result["answer"].lower()
            correctness_scores = {
                "correctness": 0.0 if answered else 1.0,
                "relevance": 1.0,
                "topic_coverage": 1.0,
                "overall": 0.0 if answered else 1.0
            }
        else:
            correctness_scores = score_answer_correctness(
                item["question"],
                result["answer"],
                item["expected_topics"]
            )
        
        # Source retrieval
        source_score = check_source_retrieval(
            result["sources"],
            item["expected_sources"]
        )
        
        eval_result = {
            "id": item["id"],
            "question": item["question"],
            "difficulty": item["difficulty"],
            "answer": result["answer"],
            "correctness_scores": correctness_scores,
            "source_retrieval_score": source_score,
            "confidence": result["confidence"],
            "citations_verified": len(result["citations"]),
            "citations_supported": sum(
                1 for v in result["citations"].values()
                if v.get("supported", False)
            )
        }
        
        results.append(eval_result)
        
        print(f"  Correctness: {correctness_scores['overall']:.2f} | "
              f"Sources: {source_score:.2f} | "
              f"Confidence: {result['confidence']['composite_score']:.2f}")
    
    # Aggregate metrics
    avg_correctness = sum(r["correctness_scores"]["overall"] for r in results) / len(results)
    avg_source_retrieval = sum(r["source_retrieval_score"] for r in results) / len(results)
    avg_confidence = sum(r["confidence"]["composite_score"] for r in results) / len(results)
    
    total_citations = sum(r["citations_verified"] for r in results)
    total_supported = sum(r["citations_supported"] for r in results)
    citation_accuracy = total_supported / total_citations if total_citations > 0 else 0

    summary = {
        "total_questions": len(results),
        "avg_correctness": round(avg_correctness, 3),
        "avg_source_retrieval": round(avg_source_retrieval, 3),
        "avg_confidence": round(avg_confidence, 3),
        "citation_accuracy": round(citation_accuracy, 3),
        "results": results
    }
    
    # Save results
    output_path = "data/eval_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions:      {summary['total_questions']}")
    print(f"Avg correctness:      {summary['avg_correctness']:.1%}")
    print(f"Avg source retrieval: {summary['avg_source_retrieval']:.1%}")
    print(f"Avg confidence:       {summary['avg_confidence']:.1%}")
    print(f"Citation accuracy:    {summary['citation_accuracy']:.1%}")
    print(f"\nResults saved to {output_path}")
    
    return summary


if __name__ == "__main__":
    # Run on first 10 questions to save API costs
    # Change to 30 for full eval
    summary = run_evaluation(sample_size=10)