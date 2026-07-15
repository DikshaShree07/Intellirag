# IntelliRAG – Agentic Retrieval-Augmented Generation System

An intelligent Retrieval-Augmented Generation (RAG) application built using Python, ChromaDB, Google Gemini, Streamlit, and CLIP. The system retrieves relevant information from local documents, generates grounded responses using an LLM, and falls back to live web search when local knowledge is insufficient.

---

## Features

- Semantic document retrieval using Sentence Transformers
- Local vector database powered by ChromaDB
- LLM-powered answer generation using Google Gemini
- Automatic fallback to Tavily Web Search when local context is insufficient
- Multimodal support through CLIP-based image embeddings
- Interactive Streamlit web interface
- PDF document ingestion and indexing
- Image ingestion and similarity search
- Local knowledge base with vector search

---

## Tech Stack

### Language
- Python

### Frontend
- Streamlit

### Large Language Model
- Google Gemini 2.5 Flash

### Vector Database
- ChromaDB

### Embedding Models
- all-MiniLM-L6-v2 (Sentence Transformers)
- CLIP (Image Embeddings)

### APIs
- Google Gemini API
- Tavily Search API

### Libraries
- sentence-transformers
- chromadb
- transformers
- torch
- pypdf
- Pillow
- python-dotenv

---

## Project Structure


agentic-rag/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
│
├── data/
│   ├── docs/
│   └── images/
│
├── chroma_db/
│
└── src/
    ├── ingest_text.py
    ├── ingest_images.py
    ├── pipeline.py
    ├── grader.py
    ├── rewriter.py
    └── web_search.py


---

## Workflow

1. Upload PDF documents.
2. Convert documents into embeddings.
3. Store embeddings in ChromaDB.
4. User asks a question.
5. Retrieve the most relevant document chunks.
6. Generate an answer using Google Gemini.
7. If the retrieved information is insufficient, perform a live Tavily web search.
8. Display the final response through the Streamlit interface.

---

## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/intellirag.git
cd intellirag
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file


GEMINI_API_KEY=YOUR_KEY
TAVILY_API_KEY=YOUR_KEY


---

## Run

Index your documents

```bash
python src/ingest_text.py
```

(Optional)

```bash
python src/ingest_images.py
```

Launch the application

```bash
streamlit run app.py
```

---

## Sample Questions

- What is Machine Learning?
- Explain Gradient Descent.
- Summarize this research paper.
- What are the differences between LLMs and SLMs?
- What is Agentic AI?

---

## Future Improvements

- OCR support for scanned PDFs
- Hybrid search (Keyword + Semantic)
- Better retrieval evaluation
- Docker deployment
- User authentication
- Cloud-hosted vector database
- Improved multimodal reasoning





