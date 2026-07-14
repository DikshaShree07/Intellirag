"""
Loads PDFs/text files from data/docs, splits them into chunks,
embeds them with a free sentence-transformers model, and stores
them in a local ChromaDB collection called "text_chunks".

Run this once (or whenever you add new documents):
    python src/ingest_text.py
"""

import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "docs")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap between chunks so context isn't cut off


def read_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def main():
    print("Loading embedding model (first run downloads ~90MB, then it's cached)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection("text_chunks")

    files = glob.glob(os.path.join(DOCS_DIR, "*.pdf")) + glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    if not files:
        print(f"No .pdf or .txt files found in {DOCS_DIR}. Add some documents first.")
        return

    all_chunks, all_ids, all_metadatas = [], [], []
    for path in files:
        text = read_pdf(path) if path.endswith(".pdf") else read_txt(path)
        chunks = chunk_text(text)
        fname = os.path.basename(path)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{fname}_{i}")
            all_metadatas.append({"source": fname, "chunk_index": i})
        print(f"  {fname}: {len(chunks)} chunks")

    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    collection.upsert(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadatas,
    )
    print(f"Done. Stored {len(all_chunks)} text chunks in ChromaDB at {DB_DIR}")


if __name__ == "__main__":
    main()
