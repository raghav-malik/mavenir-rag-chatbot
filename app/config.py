import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- Embedding Model ---
# Telecom-domain fine-tuned model: +9.6 to +60.2 NDCG@10 over general-purpose models
# Source: https://huggingface.co/farbodtavakkoli/OTel-Embedding-34M
EMBEDDING_MODEL = "farbodtavakkoli/OTel-Embedding-34M"
EMBEDDING_DIMENSION = 384

# --- Reranker ---
# Cross-encoder for two-phase retrieval (Chat3GPP approach)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# --- Chunking ---
# 512 tokens balances context retention vs noise for technical docs
# 50-token overlap prevents information loss at chunk boundaries
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# --- Retrieval ---
# Retrieve 20 candidates via hybrid search, rerank to top 5
RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 5

# --- LLM ---
DEFAULT_LLM_PROVIDER = "openai"  # "openai" or "anthropic"
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# --- ChromaDB ---
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "telecom_3gpp"

# --- Data ---
DATA_DIR = "./data"