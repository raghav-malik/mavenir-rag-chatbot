from app.retriever import retrieve
from app.llm_adapter import call_llm
from app.faithfulness import verify_faithfulness


SYSTEM_PROMPT = """You are a 3GPP telecommunications standards expert assistant.

STRICT RULES:
1. Answer ONLY using the provided context. Do not use any prior knowledge.
2. For every claim you make, cite the source document in [brackets].
3. If the context does not contain enough information to answer the question, 
   respond with: "I don't have enough information in the available documents to answer this question."
4. Do not speculate or infer beyond what the context explicitly states.
5. Keep answers clear, technical, and precise.

These rules exist to ensure zero hallucination. Every statement must be traceable 
to a specific source document."""


def build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block.
    
    Each chunk is labeled with its source document so the LLM
    can cite specific documents in its response.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"]["source"]
        text = chunk["text"]
        context_parts.append(f"[Source {i}: {source}]\n{text}")
    
    return "\n\n---\n\n".join(context_parts)


def build_user_message(query: str, context: str) -> str:
    """Combine context and query into the user message."""
    return f"""Context from 3GPP specifications:

{context}

---

Question: {query}

Answer the question using ONLY the context above. Cite sources using [Source N] format."""


def classify_confidence(response: str, chunks: list[dict]) -> str:
    """
    Classify whether the response was grounded in context or not.
    
    Simple heuristic: if the response contains our "insufficient info"
    phrase, it's NOT_GROUNDED. Otherwise, check if source citations exist.
    """
    if "don't have enough information" in response.lower():
        return "NOT_GROUNDED"
    
    has_citations = any(f"[Source {i}" in response for i in range(1, len(chunks) + 1))
    
    if has_citations:
        return "GROUNDED_WITH_CITATIONS"
    else:
        return "GROUNDED_NO_CITATIONS"


def generate(query: str, provider: str = None) -> dict: # type: ignore
    """
    Full RAG generation pipeline with faithfulness verification:
    1. Retrieve top 5 relevant chunks
    2. Build grounded prompt with context
    3. Call LLM with strict grounding instructions
    4. Classify confidence level
    5. Verify faithfulness of claims against context
    6. Return answer with sources, confidence, and faithfulness
    """
    # Step 1: Retrieve relevant chunks
    chunks = retrieve(query)
    
    # Step 2: Build the prompt
    context = build_context(chunks)
    user_message = build_user_message(query, context)
    
    # Step 3: Call the LLM
    kwargs = {}
    if provider:
        kwargs["provider"] = provider
    response = call_llm(SYSTEM_PROMPT, user_message, **kwargs)
    
    # Step 4: Classify confidence
    confidence = classify_confidence(response, chunks)
    
    # Step 5: Verify faithfulness (only if the system answered)
    faithfulness_result = None
    if confidence != "NOT_GROUNDED":
        faithfulness_result = verify_faithfulness(response, chunks, provider)
        
        # If faithfulness check finds unsupported claims, downgrade confidence
        if faithfulness_result and not faithfulness_result["is_faithful"]:
            confidence = "PARTIALLY_GROUNDED"
    
    # Step 6: Package the result
    sources = [
        {
            "source": chunk["metadata"]["source"],
            "chunk_index": chunk["metadata"]["chunk_index"],
            "relevance_score": chunk["rerank_score"],
            "text_preview": chunk["text"][:200]
        }
        for chunk in chunks
    ]
    
    result = {
        "query": query,
        "answer": response,
        "confidence": confidence,
        "sources": sources,
        "model_provider": provider or "default"
    }
    
    if faithfulness_result:
        result["faithfulness"] = faithfulness_result
    
    return result