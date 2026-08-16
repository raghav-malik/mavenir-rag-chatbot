from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.generator import generate
from app.ingest import ingest
import os
from app.config import DATA_DIR
from typing import Literal, Optional

app = FastAPI(
    title="3GPP RAG Chatbot",
    description="RAG chatbot for 3GPP telecom standards with near-zero hallucination",
    version="1.0.0"
)


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

@app.get("/")
def root():
    return {
        "name": "3GPP RAG Chatbot",
        "description": "Retrieval-Augmented Generation for 3GPP Telecom Standards",
        "endpoints": {
            "POST /ask": "Ask a question about 3GPP standards",
            "POST /ingest": "Ingest documents from data directory",
            "GET /sources": "List ingested documents"
        }
    }


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Ask a question about 3GPP telecom standards.
    
    The pipeline:
    1. Embed the question using OTel telecom embedding model
    2. Retrieve top 20 candidates via semantic search (ChromaDB)
    3. Rerank to top 5 using cross-encoder (ms-marco-MiniLM-L6-v2)
    4. Generate grounded answer with citations via LLM
    5. Classify confidence level
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if len(request.question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long (max 1000 characters)")
    
    result = generate(request.question, provider=request.provider) # type: ignore
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


@app.get("/sources")
def list_sources():
    """List all documents available in the data directory."""
    files = [f for f in os.listdir(DATA_DIR) if f.endswith((".docx", ".pdf"))]
    return {
        "total_documents": len(files),
        "documents": files
    }