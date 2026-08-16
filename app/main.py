import time
import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from app.generator import generate
from app.ingest import ingest
import os
from app.config import DATA_DIR, OPENAI_API_KEY, ANTHROPIC_API_KEY, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from typing import Literal, Optional
from collections import deque
from datetime import datetime, timezone
import chromadb

query_log = deque(maxlen=50)

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("rag-chatbot")

app = FastAPI(
    title="3GPP RAG Chatbot",
    description="RAG chatbot for 3GPP telecom standards with near-zero hallucination",
    version="1.0.0"
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000)
    logger.info(f"{request.method} {request.url.path} {response.status_code} {duration_ms}ms")
    return response


class QuestionRequest(BaseModel):
    question: str
    provider: Optional[Literal["openai", "anthropic"]] = None  # type: ignore


class AnswerResponse(BaseModel):
    query: str
    answer: str
    confidence: str
    sources: list
    model_provider: str
    faithfulness: dict = None # type: ignore
    latency_ms: int = None # type: ignore

@app.get("/")
def root():
    return {
        "name": "3GPP RAG Chatbot",
        "description": "Retrieval-Augmented Generation for 3GPP Telecom Standards",
        "endpoints": {
            "POST /ask": "Ask a question about 3GPP standards",
            "POST /ingest": "Ingest documents from data directory",
            "GET /sources": "List ingested documents",
            "GET /health": "System health: ChromaDB status, providers, BM25 index",
            "GET /history": "Last 50 queries with confidence and latency"
        }
    }


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Ask a question about 3GPP telecom standards.
    
    Pipeline: query expansion → hybrid retrieval (BM25 + semantic) → 
    cross-encoder rerank → grounded generation → faithfulness verification → 
    confidence classification
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if len(request.question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long (max 1000 characters)")
    
    start_time = time.time()
    result = generate(request.question, provider=request.provider) # type: ignore
    result["latency_ms"] = round((time.time() - start_time) * 1000)

    query_log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": request.question,
        "confidence": result["confidence"],
        "faithfulness_score": result.get("faithfulness", {}).get("faithfulness") if result.get("faithfulness") else None,
        "latency_ms": result["latency_ms"]
    })

    return result


@app.post("/ingest")
def ingest_documents():
    """Ingest all documents from the data directory.
    
    Loads PDFs and DOCX files, chunks them, generates embeddings
    using the OTel telecom model, and stores in ChromaDB.
    """
    files = [f for f in os.listdir(DATA_DIR) if f.endswith((".docx", ".pdf"))]
    
    if not files:
        raise HTTPException(status_code=404, detail="No documents found in data directory")
    
    ingest()
    
    # Reset the retriever so it rebuilds BM25 index with new data
    import app.retriever as retriever_module
    retriever_module._retriever = None
    
    return {
        "status": "success",
        "message": f"Ingested {len(files)} documents",
        "files": files
    }


@app.get("/health")
def health_check():
    """System health: ChromaDB status, available providers, BM25 index state."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        chunk_count = collection.count()
    except Exception:
        chunk_count = 0

    providers = []
    if OPENAI_API_KEY:
        providers.append("openai")
    if ANTHROPIC_API_KEY:
        providers.append("anthropic")

    import app.retriever as retriever_module
    bm25_loaded = retriever_module._retriever is not None

    return {
        "status": "healthy",
        "chunks_indexed": chunk_count,
        "available_providers": providers,
        "bm25_index_loaded": bm25_loaded
    }


@app.get("/history")
def query_history():
    """Last 50 queries with confidence, faithfulness, and latency."""
    return {
        "total_logged": len(query_log),
        "queries": list(query_log)
    }


@app.get("/sources")
def list_sources():
    """List all documents available in the data directory."""
    files = [f for f in os.listdir(DATA_DIR) if f.endswith((".docx", ".pdf"))]
    return {
        "total_documents": len(files),
        "documents": files
    }