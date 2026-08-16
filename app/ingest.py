import os
from docx import Document as DocxDocument
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from app.config import (
    EMBEDDING_MODEL, DATA_DIR, CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
)


def load_docx(file_path: str) -> str:
    """Extract text from a .docx file."""
    doc = DocxDocument(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


def load_pdf(file_path: str) -> str:
    """Extract text from a .pdf file."""
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])


def load_document(file_path: str) -> str:
    """Load a document based on its extension."""
    if file_path.endswith(".docx"):
        return load_docx(file_path)
    elif file_path.endswith(".pdf"):
        return load_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into chunks of roughly chunk_size characters with overlap.
    
    800 chars (~200 tokens) stays within OTel-Embedding-34M's 1500-token limit.
    Telco-RAG/Telco-oRAG research shows smaller focused chunks outperform larger ones.
    80 char overlap prevents information loss at chunk boundaries.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def load_all_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all documents from the data directory and chunk them."""
    all_chunks = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith((".docx", ".pdf")):
            file_path = os.path.join(data_dir, filename)
            print(f"Loading: {filename}")
            
            text = load_document(file_path)
            chunks = chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": f"{filename}__chunk_{i}",
                    "text": chunk,
                    "metadata": {
                        "source": filename,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                })
            
            print(f"  → {len(chunks)} chunks created from {filename}")
    
    print(f"\nTotal: {len(all_chunks)} chunks from {len([f for f in os.listdir(data_dir) if f.endswith(('.docx', '.pdf'))])} files")
    return all_chunks


def create_embeddings_and_store(chunks: list[dict]):
    """Generate embeddings using OTel model and store in ChromaDB."""
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Extract texts for embedding
    texts = [chunk["text"] for chunk in chunks]
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    
    # Store in ChromaDB
    print(f"\nStoring in ChromaDB at {CHROMA_PERSIST_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    # Delete existing collection if it exists (clean re-ingestion)
    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        pass
    
    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"description": "3GPP Telecom Standards for RAG"}
    )
    
    # ChromaDB has a batch limit, so we add in batches of 500
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        collection.add(
            ids=[chunk["id"] for chunk in chunks[i:batch_end]],
            documents=[chunk["text"] for chunk in chunks[i:batch_end]],
            embeddings=embeddings[i:batch_end].tolist(),
            metadatas=[chunk["metadata"] for chunk in chunks[i:batch_end]]
        )
    
    print(f"✓ Stored {len(chunks)} chunks in collection '{CHROMA_COLLECTION_NAME}'")
    return collection


def ingest():
    """Full ingestion pipeline: load → chunk → embed → store."""
    chunks = load_all_documents()
    if not chunks:
        print("No documents found in data directory.")
        return
    collection = create_embeddings_and_store(chunks)
    return collection


if __name__ == "__main__":
    ingest()