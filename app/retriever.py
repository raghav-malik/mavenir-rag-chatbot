from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from app.config import (
    EMBEDDING_MODEL, RERANKER_MODEL, CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME, RETRIEVAL_TOP_K, RERANK_TOP_K
)

# Load models once at module level so they don't reload on every query
print("Loading retrieval models...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
reranker = CrossEncoder(RERANKER_MODEL)
print("Retrieval models loaded.")


def get_collection():
    """Connect to existing ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_collection(name=CHROMA_COLLECTION_NAME)


def semantic_search(query: str, collection, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """
    Stage 1: Embed the query and find similar chunks in ChromaDB.
    
    This converts the user's question into 384 numbers using the same
    OTel model we used during ingestion, then finds the chunks whose
    embeddings point in the most similar direction (cosine similarity).
    """
    query_embedding = embedder.encode(query, normalize_embeddings=True).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Unpack ChromaDB's nested list format
    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
    
    return chunks


def rerank(query: str, chunks: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
    """
    Stage 2: Re-score each chunk using the cross-encoder.
    
    Semantic search (Stage 1) is fast but approximate — it encodes query
    and chunk separately, then compares. The cross-encoder is slower but
    more accurate — it reads the query and chunk TOGETHER as one input,
    so it understands the relationship between them.
    
    Think of it like this:
    - Stage 1 (bi-encoder): "Does this chunk LOOK like it's about the same topic?"
    - Stage 2 (cross-encoder): "Does this chunk actually ANSWER this question?"
    """
    # Create (query, chunk) pairs for the cross-encoder
    pairs = [(query, chunk["text"]) for chunk in chunks]
    
    # Score each pair
    scores = reranker.predict(pairs)
    
    # Attach scores to chunks
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])
    
    # Sort by rerank score (highest = most relevant) and take top_k
    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return chunks[:top_k]


def retrieve(query: str) -> list[dict]:
    """
    Full retrieval pipeline: semantic search → rerank → top 5 chunks.
    
    This is the three-stage approach from Chat3GPP research:
    1. Cast a wide net: get 20 candidates via semantic similarity
    2. Narrow it down: cross-encoder reranks to find the 5 best
    3. (Stage 3 is in generator.py — feed these to the LLM)
    """
    collection = get_collection()
    
    # Stage 1: Get 20 candidates via semantic search
    candidates = semantic_search(query, collection)
    
    # Stage 2: Rerank to top 5
    top_chunks = rerank(query, candidates)
    
    return top_chunks