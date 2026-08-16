import json
import re
import os
from typing import Optional


def load_glossary(path: Optional[str] = None) -> dict: 
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "glossary.json")
    with open(path, "r") as f:
        return json.load(f)


GLOSSARY = load_glossary()


def expand_query(query: str, glossary: Optional[dict] = None) -> str:
    """
    Expand telecom acronyms in the query to improve retrieval.
    
    Example:
      "What does AMF do?" 
      → "What does AMF do? | AMF Access and Mobility Management Function"
    
    Why: 3GPP specs sometimes use full names instead of acronyms.
    Semantic search on "AMF" might miss chunks that say 
    "Access and Mobility Management Function" without the acronym.
    This is Telco-RAG's glossary-based query expansion approach.
    """
    if glossary is None:
        glossary = GLOSSARY
    
    expansions = []
    for acronym, full_name in glossary.items():
        if re.search(rf"\b{re.escape(acronym)}\b", query):
            expansions.append(f"{acronym} {full_name}")
    
    if expansions:
        return query + " | " + " | ".join(expansions)
    return query