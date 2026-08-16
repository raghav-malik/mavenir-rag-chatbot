import os
from docx import Document as DocxDocument
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from app.config import (
    EMBEDDING_MODEL, DATA_DIR, CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
)


def is_heading(paragraph) -> bool:
    """Check if a paragraph is a heading based on its style."""
    style_name = paragraph.style.name.lower() if paragraph.style else ""
    return "heading" in style_name


def get_heading_level(paragraph) -> int:
    """Extract heading level (1, 2, 3...) from style name."""
    style_name = paragraph.style.name.lower() if paragraph.style else ""
    for i in range(1, 10):
        if f"heading {i}" in style_name:
            return i
    return 0


def parse_docx_by_sections(file_path: str) -> list[dict]:
    """
    Parse a DOCX file into sections based on heading structure.
    
    Instead of blindly cutting at character boundaries, this reads
    the document structure — headings, subheadings, body text —
    and groups text under its parent heading.
    
    Why this matters for 3GPP: specifications are hierarchical.
    Clause 5.4.4.1 has a heading and body text that belong together.
    Character-based chunking might split them across two chunks.
    """
    doc = DocxDocument(file_path)
    sections = []
    
    current_heading = "Introduction"
    current_heading_level = 0
    current_breadcrumb = []
    current_text = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        
        if is_heading(para):
            # Save the previous section if it has content
            if current_text:
                sections.append({
                    "heading": current_heading,
                    "breadcrumb": " > ".join(current_breadcrumb) if current_breadcrumb else current_heading,
                    "text": "\n".join(current_text),
                    "heading_level": current_heading_level
                })
            
            # Update heading tracking
            level = get_heading_level(para)
            current_heading = text
            current_heading_level = level
            
            # Update breadcrumb (keep parent headings, replace at current level)
            if level <= len(current_breadcrumb):
                current_breadcrumb = current_breadcrumb[:level - 1]
            current_breadcrumb.append(text)
            
            current_text = []
        else:
            current_text.append(text)
    
    # Don't forget the last section
    if current_text:
        sections.append({
            "heading": current_heading,
            "breadcrumb": " > ".join(current_breadcrumb) if current_breadcrumb else current_heading,
            "text": "\n".join(current_text),
            "heading_level": current_heading_level
        })
    
    return sections


def chunk_section(section: dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Chunk a section into smaller pieces if it's too long.
    
    Short sections (under chunk_size) stay whole — no splitting.
    Long sections get split with overlap, but the heading/breadcrumb
    metadata stays attached to every chunk.
    """
    text = section["text"]
    heading = section["heading"]
    breadcrumb = section["breadcrumb"]
    
    # If section fits in one chunk, keep it whole
    if len(text) <= chunk_size:
        return [{
            "text": text,
            "heading": heading,
            "breadcrumb": breadcrumb
        }]
    
    # Otherwise split with overlap
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "heading": heading,
                "breadcrumb": breadcrumb
            })
        start = end - overlap
    
    return chunks


def load_and_chunk_docx(file_path: str) -> list[dict]:
    """Full pipeline for one DOCX: parse sections → chunk → tag metadata."""
    filename = os.path.basename(file_path)
    sections = parse_docx_by_sections(file_path)
    
    all_chunks = []
    for section in sections:
        section_chunks = chunk_section(section)
        for i, chunk in enumerate(section_chunks):
            all_chunks.append({
                "id": f"{filename}__chunk_{len(all_chunks)}",
                "text": chunk["text"],
                "metadata": {
                    "source": filename,
                    "heading": chunk["heading"],
                    "breadcrumb": chunk["breadcrumb"],
                    "chunk_index": len(all_chunks),
                    "total_chunks": 0  # filled in later
                }
            })
    
    # Fill in total_chunks
    for chunk in all_chunks:
        chunk["metadata"]["total_chunks"] = len(all_chunks)
    
    return all_chunks


def load_pdf_and_chunk(file_path: str) -> list[dict]:
    """Fallback for PDFs — character-based chunking (no heading structure)."""
    filename = os.path.basename(file_path)
    reader = PdfReader(file_path)
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "id": f"{filename}__page_chunk__{len(chunks)}",
                "text": chunk_text,
                "metadata": {
                    "source": filename,
                    "heading": "unknown",
                    "breadcrumb": filename,
                    "chunk_index": len(chunks),
                    "total_chunks": 0
                }
            })
        start = end - CHUNK_OVERLAP
    
    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = len(chunks)
    
    return chunks


def load_all_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all documents with clause-aware chunking for DOCX."""
    all_chunks = []
    doc_files = [f for f in os.listdir(data_dir) if f.endswith((".docx", ".pdf"))]
    
    for filename in doc_files:
        file_path = os.path.join(data_dir, filename)
        print(f"Loading: {filename}")
        
        if filename.endswith(".docx"):
            chunks = load_and_chunk_docx(file_path)
        else:
            chunks = load_pdf_and_chunk(file_path)
        
        print(f"  → {len(chunks)} chunks created from {filename}")
        all_chunks.extend(chunks)
    
    print(f"\nTotal: {len(all_chunks)} chunks from {len(doc_files)} files")
    return all_chunks


def create_embeddings_and_store(chunks: list[dict]):
    """Generate embeddings using OTel model and store in ChromaDB."""
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    texts = [chunk["text"] for chunk in chunks]
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    
    print(f"\nStoring in ChromaDB at {CHROMA_PERSIST_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        pass
    
    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"description": "3GPP Telecom Standards for RAG (clause-aware chunking)"}
    )
    
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
    """Full ingestion pipeline: parse sections → chunk → embed → store."""
    chunks = load_all_documents()
    if not chunks:
        print("No documents found in data directory.")
        return
    collection = create_embeddings_and_store(chunks)
    return collection


if __name__ == "__main__":
    ingest()