from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
import re
from app.config import (
    EMBEDDING_MODEL, RERANKER_MODEL, CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME, RETRIEVAL_TOP_K, RERANK_TOP_K
)
from app.query_expander import expand_query

# Load models once at module level
print("Loading retrieval models...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
reranker = CrossEncoder(RERANKER_MODEL)
print("Retrieval models loaded.")

# Tokenizer for BM25 — keeps telecom identifiers like S-NSSAI, 5GC, N2
_token_pattern = re.compile(r"[A-Za-z0-9_./+-]+")

def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _token_pattern.findall(text)]


class HybridRetriever:
    """
    Three-stage retrieval pipeline:
    
    Stage 1a: Dense semantic search (ChromaDB + OTel embeddings)
      → finds conceptually similar chunks
    Stage 1b: Sparse keyword search (BM25)
      → finds exact term matches (spec numbers, acronyms)
    Stage 1c: Reciprocal Rank Fusion
      → merges both ranked lists into one
    Stage 2: Cross-encoder reranking
      → reads query + chunk together for precise relevance scoring
    
    Why hybrid? 3GPP docs are dense with exact identifiers (S-NSSAI, 
    clause 5.4.4.1, AMF). Semantic search alone misses exact matches.
    BM25 alone misses conceptual matches. Hybrid catches both.
    This is the approach validated by Chat3GPP (arXiv:2501.13954).
    """
    
    def __init__(self):
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        
        # Load all chunks for BM25 index
        print("Building BM25 index...")
        all_data = self.collection.get(include=["documents", "metadatas"])
        
        self.chunk_ids = all_data["ids"]
        self.chunk_texts = all_data["documents"]
        self.chunk_metadatas = all_data["metadatas"]
        
        # Build lookup dict
        self.chunks_by_id = {}
        for i in range(len(self.chunk_ids)):
            self.chunks_by_id[self.chunk_ids[i]] = {
                "id": self.chunk_ids[i],
                "text": self.chunk_texts[i], # type: ignore
                "metadata": self.chunk_metadatas[i] # type: ignore
            }
        
        # Build BM25 index over all chunks
        tokenized_corpus = [tokenize(text) for text in self.chunk_texts] # type: ignore
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index built over {len(self.chunk_ids)} chunks.")
    
    def dense_search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[tuple]:
        """
        Semantic search via ChromaDB.
        Returns list of (chunk_id, distance) sorted by ascending distance.
        """
        query_embedding = embedder.encode(query, normalize_embeddings=True).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["distances"]
        )
        
        # Return as (id, distance) pairs — lower distance = more similar
        pairs = list(zip(results["ids"][0], results["distances"][0])) # type: ignore
        pairs.sort(key=lambda x: x[1])  # ascending distance
        return pairs
    
    def bm25_search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[tuple]:
        """
        Keyword search via BM25.
        Returns list of (chunk_id, score) sorted by descending score.
        """
        scores = self.bm25.get_scores(tokenize(query))
        
        # Pair scores with chunk IDs and sort descending
        paired = list(zip(self.chunk_ids, scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired[:top_k]
    
    def reciprocal_rank_fusion(
        self, 
        ranked_lists: list[list[tuple]], 
        k: int = 60, 
        top_n: int = RETRIEVAL_TOP_K
    ) -> list[str]:
        """
        Merge multiple ranked lists using Reciprocal Rank Fusion.
        
        RRF score = sum over all lists of: 1 / (k + rank)
        
        Why RRF over simple score averaging?
        - Scores from different systems aren't comparable 
          (ChromaDB distance vs BM25 score)
        - RRF only uses RANK position, which is universal
        - k=60 is the standard constant from the original RRF paper
        """
        fused_scores = {}
        
        for ranked_list in ranked_lists:
            for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0.0
                fused_scores[doc_id] += 1.0 / (k + rank)
        
        # Sort by fused score descending
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _score in sorted_results[:top_n]]
    
    def rerank(self, query: str, chunks: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
        """
        Stage 2: Cross-encoder reranking.
        Reads query and chunk TOGETHER for precise relevance scoring.
        """
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = reranker.predict(pairs)
        
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[i])
        
        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return chunks[:top_k]


# Global retriever instance — loaded once
_retriever = None

def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        try:
            _retriever = HybridRetriever()
        except Exception as e:
            raise RuntimeError("No documents ingested. Run 'python -m app.ingest' first.") from e
    return _retriever


def retrieve(query: str) -> list[dict]:
    """
    Full hybrid retrieval pipeline:
    1. Expand query with telecom glossary
    2. Run dense (semantic) + sparse (BM25) search in parallel
    3. Fuse results with Reciprocal Rank Fusion → top 20
    4. Cross-encoder rerank → top 5
    """
    retriever = get_retriever()
    
    # Step 0: Expand acronyms
    expanded_query = expand_query(query)
    
    # Step 1a: Dense semantic search
    dense_results = retriever.dense_search(expanded_query)
    
    # Step 1b: BM25 keyword search
    bm25_results = retriever.bm25_search(expanded_query)
    
    # Step 1c: Fuse with RRF
    fused_ids = retriever.reciprocal_rank_fusion([dense_results, bm25_results])
    
    # Convert IDs back to chunk dicts
    fused_chunks = [retriever.chunks_by_id[cid] for cid in fused_ids if cid in retriever.chunks_by_id]
    
    # Step 2: Cross-encoder rerank
    top_chunks = retriever.rerank(query, fused_chunks)
    
    return top_chunks