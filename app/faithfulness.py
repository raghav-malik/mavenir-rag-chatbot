import json
import re
from typing import Optional
from app.llm_adapter import call_llm


FAITHFULNESS_PROMPT = """You are a strict fact checker. Your job is to verify whether 
an answer is fully supported by the provided context.

Given CONTEXT and ANSWER:
1. Extract every factual claim from the ANSWER
2. For each claim, check if it is fully supported by the CONTEXT
3. A claim is "supported" ONLY if the CONTEXT explicitly states it
4. If a claim adds information not in the CONTEXT, mark it unsupported

Return ONLY valid JSON, no other text:
{{"claims": [{{"claim": "...", "supported": true}}, {{"claim": "...", "supported": false}}], "faithfulness": 0.0}}

faithfulness = number of supported claims / total claims (0.0 to 1.0)

CONTEXT:
{context}

ANSWER:
{answer}"""


def extract_json(text: str) -> dict:
    """Safely extract JSON from LLM response."""
    try:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        pass
    return {"claims": [], "faithfulness": 0.0}


def verify_faithfulness(answer: str, chunks: list[dict], provider: Optional[str] = None) -> dict:
    """
    Post-generation faithfulness check.
    
    Takes the generated answer and the retrieved chunks, asks a second
    LLM call to verify: "Is every claim in this answer actually supported
    by the retrieved context?"
    
    This catches "citation theater" — when the model writes [Source 1] 
    but the claim isn't actually in Source 1.
    
    Returns:
        faithfulness: float (0.0 to 1.0)
        is_faithful: bool (True if all claims supported)
        unsupported_claims: list of claims not in context
    """
    # Build context from chunks
    context = "\n\n".join(
        f"[Source {i+1}: {c['metadata']['source']}]\n{c['text']}" 
        for i, c in enumerate(chunks)
    )
    
    # Ask LLM to verify
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    
    raw_response = call_llm(
        "You are a JSON-only fact checker. Return only valid JSON.", 
        prompt, 
        provider=provider or "anthropic"
    )
    
    data = extract_json(raw_response)
    
    claims = data.get("claims", [])
    supported = [c for c in claims if c.get("supported") is True]
    unsupported = [c.get("claim", "") for c in claims if not c.get("supported")]
    
    score = len(supported) / max(len(claims), 1)
    
    return {
        "faithfulness": round(score, 3),
        "is_faithful": len(unsupported) == 0 and len(claims) > 0,
        "total_claims": len(claims),
        "supported_claims": len(supported),
        "unsupported_claims": unsupported
    }