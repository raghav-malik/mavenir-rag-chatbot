# 3GPP RAG Chatbot

A Retrieval-Augmented Generation chatbot for 3GPP telecom standards with a focus on faithful, abstention-aware answers. Built as a project submission for Mavenir's Graduate Engineer Trainee – AI/LLM Engineer (Telecom AI Ops) position.

## Architecture

```
User Question
       │
       ▼
┌──────────────────┐
│  Query Expansion │  Expands telecom acronyms using 3GPP glossary
│  (glossary.json) │  "AMF" → "AMF Access and Mobility Management Function"
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│           STAGE 1: HYBRID RETRIEVAL              │
│                                                  │
│  ┌─────────────┐        ┌─────────────┐         │
│  │   Semantic   │        │    BM25     │         │
│  │   Search     │        │   Keyword   │         │
│  │  (ChromaDB)  │        │   Search    │         │
│  │  OTel-34M    │        │ (rank_bm25) │         │
│  └──────┬──────┘        └──────┬──────┘         │
│         │    Top 20            │   Top 20        │
│         └─────────┬────────────┘                 │
│                   ▼                              │
│         ┌─────────────────┐                      │
│         │ Reciprocal Rank │  Merges both lists   │
│         │     Fusion      │  using rank position │
│         └────────┬────────┘                      │
│                  │ Top 20 fused                  │
└──────────────────┼───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│        STAGE 2: CROSS-ENCODER RERANKING          │
│                                                  │
│  ms-marco-MiniLM-L6-v2 reads each (query, chunk)│
│  pair together and scores relevance              │
│                                                  │
│  Top 20 → Top 5 most relevant                   │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│        STAGE 3: GROUNDED GENERATION              │
│                                                  │
│  System prompt enforces:                         │
│  • Answer ONLY from provided context             │
│  • Cite every claim with [Source N]              │
│  • Say "I don't know" if context insufficient    │
│                                                  │
│  LLM: OpenAI or Anthropic (adapter pattern)      │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│     STAGE 4: FAITHFULNESS VERIFICATION           │
│                                                  │
│  LLM-as-judge extracts claims from the answer    │
│  and verifies each against retrieved context     │
│                                                  │
│  Catches "citation theater" — when the model     │
│  writes [Source 1] but the claim isn't in it     │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│       CONFIDENCE CLASSIFICATION                  │
│                                                  │
│  GROUNDED_WITH_CITATIONS  — cited + verified     │
│  PARTIALLY_GROUNDED       — cited but claims     │
│                             not fully supported  │
│  GROUNDED_NO_CITATIONS    — answered, no cites   │
│  NOT_GROUNDED             — honest "I don't know"│
└──────────────────────────────────────────────────┘
```

## Evaluation Results

### Answerable Questions (10 queries about 3GPP standards)

| Metric | Result | What It Measures |
|---|---|---|
| Source Accuracy | **100%** | Retrieval always found the correct specification |
| Answer Rate | **100%** | System answered every answerable question |
| Grounding Rate | **80%** | 8/10 answers included source citations |
| Keyword Coverage | **68%** | Lexical overlap with expected terms |
| Faithfulness | **1.0** | All verified claims supported by retrieved context |

### Adversarial Questions (8 unanswerable queries)

| Metric | Result | What It Measures |
|---|---|---|
| Abstention Accuracy | **100%** | Correctly refused every unanswerable question |
| False Answers | **0/8** | Never hallucinated on out-of-scope queries |

Adversarial queries tested: WiFi 7 bandwidth (not in 3GPP), vendor product comparisons, nonexistent Release 25, financial questions, general networking (TCP), code generation requests.

### Retrieval Ablation: Dense-Only vs Hybrid

| Query: "clause 5.4.4.1 AMF" | Dense Only | Hybrid (BM25 + RRF) |
|---|---|---|
| Top result topic | AMF failover/backup | AMF authentication procedures |
| Relevant to query intent? | Partially | Yes |
| Found clause-related content? | No | Yes |

Dense-only retrieval matched on the word "AMF" appearing frequently. Hybrid retrieval, with BM25 catching the clause number and glossary expansion adding the full name, surfaced functionally relevant content.

### Comparison to Published Baselines

TSpec-LLM (arXiv:2406.01768) reported that naive RAG improved GPT-4 accuracy from 51% to 72% on 3GPP question answering. Our system achieves 100% source accuracy and 100% abstention accuracy through hybrid retrieval, cross-encoder reranking, grounded prompting, and faithfulness verification.

## Design Decisions

| Decision | Choice | Why |
|---|---|---|
| **Framework** | None (raw Python) | Every component is explainable. Submission evaluates "understanding of solution design and code implementation." |
| **Embedding Model** | OTel-Embedding-34M | Telecom-domain fine-tuned. +9.6 to +60.2 NDCG@10 over general-purpose models on telecom documents. 18M+ downloads on HuggingFace. |
| **Retrieval** | Hybrid (BM25 + Semantic + RRF) | Chat3GPP (arXiv:2501.13954) showed hybrid outperforms either alone on 3GPP docs. 3GPP specs contain both conceptual content and exact identifiers — hybrid catches both. |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L6-v2 | Bi-encoders encode query and chunk separately. Cross-encoders read them together for higher accuracy. Two-phase approach from Chat3GPP. |
| **Vector DB** | ChromaDB (persistent, local) | Runs locally with no account needed. Sufficient for project-scale corpus. Production alternative: Elasticsearch for native hybrid search. |
| **LLM Interface** | Adapter pattern (OpenAI + Anthropic) | Model-agnostic design. Swap providers via .env config. Same pattern used in production multi-model orchestration. |
| **Hallucination Reduction** | Grounded prompt + citations + faithfulness verification | Research shows retrieval grounding reduces hallucinations by 75-90%. Citation enforcement adds traceability. LLM-as-judge catches "citation theater." |
| **Evaluation** | Custom metrics + adversarial testing | Tests both accuracy (can it answer correctly?) and safety (does it refuse when it should?). Most RAG demos only test the happy path. |
| **Query Expansion** | 3GPP glossary (60+ acronyms) | Telco-RAG's approach. Expands "AMF" to include "Access and Mobility Management Function" so retrieval catches both acronym and full-name mentions. |
| **Chunk Size** | 512 characters (~100-150 tokens) | Telco-RAG found 125 tokens optimal. Telco-oRAG found 250 tokens optimal. Our size falls within the researched range. Future work: clause-aware chunking for better boundary handling. |

## Knowledge Base

Three foundational 3GPP specifications covering the 5G stack:

| Specification | Topic | Role |
|---|---|---|
| **TS 23.501** | 5G System Architecture | Defines WHAT exists: AMF, SMF, UPF, network slicing, QoS |
| **TS 23.502** | 5G System Procedures | Defines HOW things work: registration, authentication, handover |
| **TS 38.300** | NR/NG-RAN Description | Defines the radio network: gNB, NR, radio access architecture |

Total: 12,505 chunks indexed across all three specifications.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ask` | Ask a question about 3GPP standards |
| `POST` | `/ingest` | Ingest documents from the data directory |
| `GET` | `/sources` | List ingested documents |

### Example Request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is network slicing in 5G?"}'
```

### Example Response

```json
{
  "query": "What is network slicing in 5G?",
  "answer": "Network slicing in 5G refers to the capability to create multiple virtual networks on a shared physical infrastructure. Each network slice consists of a RAN part and a Core Network part, uniquely identified by S-NSSAI [Source 1]...",
  "confidence": "GROUNDED_WITH_CITATIONS",
  "sources": [
    {
      "source": "38300-j30.docx",
      "chunk_index": 412,
      "relevance_score": 6.27,
      "text_preview": "the realization of network slicing in the NG-RAN for NR connected to 5GC..."
    }
  ],
  "faithfulness": {
    "faithfulness": 1.0,
    "is_faithful": true,
    "total_claims": 5,
    "supported_claims": 5,
    "unsupported_claims": []
  }
}
```

## Project Structure

```
mavenir-rag-chatbot/
├── app/
│   ├── config.py            # All settings and design decisions
│   ├── ingest.py            # PDF/DOCX → chunking → embedding → ChromaDB
│   ├── retriever.py         # Hybrid search (BM25 + semantic + RRF) → rerank
│   ├── generator.py         # Grounded prompt → LLM → confidence classification
│   ├── llm_adapter.py       # Multi-model adapter (OpenAI + Anthropic)
│   ├── faithfulness.py      # Claim-level faithfulness verification
│   ├── query_expander.py    # Glossary-based telecom acronym expansion
│   ├── evaluation.py        # Test harness: answerable + adversarial
│   ├── glossary.json        # 60+ 3GPP acronym definitions
│   └── main.py              # FastAPI endpoints
├── data/                    # 3GPP specification documents
├── eval/
│   ├── eval_results.json    # Answerable question results
│   └── adversarial_results.json  # Adversarial question results
├── .env.example             # Required environment variables
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Setup and Installation

### Prerequisites

- Python 3.10+
- OpenAI API key and/or Anthropic API key

### Local Setup

```bash
# Clone the repository
git clone https://github.com/raghav-malik/mavenir-rag-chatbot.git
cd mavenir-rag-chatbot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys

# Ingest 3GPP documents
python -m app.ingest

# Start the server
uvicorn app.main:app --reload

# Open API documentation
# Navigate to http://localhost:8000/docs
```

### Run Evaluation

```bash
python -m app.evaluation
```

## Research References

| Paper | Key Contribution | How It Informed This Project |
|---|---|---|
| Chat3GPP (arXiv:2501.13954) | Hybrid retrieval + reranking for 3GPP | Adopted two-phase hybrid + cross-encoder approach |
| Telco-RAG (arXiv:2404.15939) | Domain-specific RAG optimization | Adopted glossary-based query expansion |
| TSpec-LLM (arXiv:2406.01768) | 3GPP evaluation baselines | Baseline comparison (GPT-4: 51% → 72% with naive RAG) |
| RAGAS (arXiv:2309.15217) | RAG evaluation framework | Informed evaluation metric design |
| OTel-Embedding (HuggingFace) | Telecom-specific embeddings | Selected as primary embedding model |

## Known Limitations and Future Work

**Current limitations:**
- Character-based chunking does not respect document structure (clause boundaries, tables). Clause-aware chunking would improve retrieval quality.
- Faithfulness verification adds latency (second LLM call per query). A lighter verification approach or caching could reduce this.
- Corpus limited to 3 specifications. Production deployment would need hundreds of specs with metadata-based pre-filtering.
- No query routing for multi-spec questions that span architecture (23.501) and procedures (23.502).

**Future enhancements:**
- Section/clause-aware chunking with hierarchical metadata (spec, clause, title)
- Metadata-filtered retrieval to scope search by specification when mentioned in query
- Reranker score threshold for pre-generation abstention
- RAGAS evaluation integration for automated faithfulness scoring at scale
- Streaming responses for better user experience
- Telco-RAG-style neural document router for multi-series retrieval
- Extension to TS 23.503 (policy), TS 29-series (APIs), TS 32-series (management)

## Tech Stack

- **Language:** Python 3.10+
- **API Framework:** FastAPI
- **Vector Database:** ChromaDB (persistent, local)
- **Embeddings:** OTel-Embedding-34M (telecom-domain fine-tuned)
- **Reranker:** cross-encoder/ms-marco-MiniLM-L6-v2
- **Keyword Search:** rank-bm25 (BM25Okapi)
- **LLM Providers:** OpenAI (GPT-4o-mini) / Anthropic (Claude Sonnet)
- **Document Parsing:** pypdf, python-docx

## Author

Raghav Malik — B.Tech Computer Science, BML Munjal University
