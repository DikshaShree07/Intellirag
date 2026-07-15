# IntelliRAG - Technical Overview

## Programming Language
- Python 3.11

---

## Frontend
- Streamlit

Purpose:
Interactive web interface for asking questions and displaying responses.

---

## Backend

Python-based Retrieval-Augmented Generation pipeline.

Main Files:
- pipeline.py
- grader.py
- rewriter.py
- web_search.py

---

## Large Language Model

Google Gemini 2.5 Flash

Purpose:
- Answer generation
- Relevance grading
- Query rewriting

---

## Vector Database

ChromaDB

Purpose:
Store semantic embeddings for document retrieval.

---

## Embedding Models

### Sentence Transformer

Model:
all-MiniLM-L6-v2

Purpose:
Convert text chunks into embeddings.

Used in:
- ingest_text.py
- pipeline.py

---

### CLIP

Purpose:
Generate image embeddings for multimodal retrieval.

Used in:
- ingest_images.py

---

## Document Processing

PyPDF

Purpose:
Extract text from PDF documents.

---

## Web Search

Tavily API

Purpose:
Retrieve live web information when local documents are insufficient.

---

## Retrieval Algorithm

Semantic Similarity Search

Database:
ChromaDB

Embedding:
Sentence Transformers

Retrieves Top-K most relevant document chunks.

---

## Query Rewriting

LLM-Based Query Reformulation

Purpose:
Improve retrieval quality when the original query is unclear.

File:
rewriter.py

---

## Relevance Grading

LLM-Based Relevance Scoring

Purpose:
Evaluate whether retrieved document chunks are useful before answer generation.

File:
grader.py

---

## Image Retrieval

CLIP Embeddings

Purpose:
Support image similarity search using multimodal embeddings.

---

## Libraries

- chromadb
- sentence-transformers
- transformers
- torch
- streamlit
- google-generativeai
- tavily-python
- pillow
- pypdf
- python-dotenv
- numpy

---

## AI Concepts Used

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Embeddings
- Vector Databases
- Transformer Models
- Multimodal AI
- Prompt Engineering
- Query Reformulation
- Relevance Ranking
- Context-Aware Generation

---

## Future Improvements

- OCR Support
- Hybrid Search (BM25 + Vector Search)
- Cross-Encoder Re-ranking
- FAISS Integration
- LangGraph Agent Workflow
- Docker Deployment
- Authentication