# Agentic Self-Correcting RAG (with Computer Vision)

A RAG system that doesn't blindly trust its own retrieval. It grades whether
retrieved chunks actually answer the question, and if not, it either rewrites
the query and tries again, or falls back to a live web search — instead of
letting the LLM hallucinate from irrelevant context.

It also supports image-based retrieval using CLIP (a pretrained computer
vision model), so you can eventually extend it to answer questions about
diagrams/charts, not just text.

## Architecture

1. **Retrieve** — question is embedded and matched against your document
   chunks in ChromaDB.
2. **Grade** — Gemini checks whether each retrieved chunk is actually
   relevant.
3. **Correct** — if nothing relevant was found: rewrite the query and
   retry once, then fall back to a live web search (Tavily) if it still
   fails.
4. **Answer** — Gemini generates a final answer grounded only in whatever
   context passed the relevance check.

## Project structure

```
agentic-rag/
├── app.py                 # Streamlit demo UI
├── requirements.txt
├── .env.example           # copy to .env and fill in your keys
├── data/
│   ├── docs/               # put your PDFs/.txt files here
│   └── images/             # put images here for CLIP retrieval
├── chroma_db/              # auto-created, stores your local vector database
└── src/
    ├── ingest_text.py       # run once to embed your documents
    ├── ingest_images.py     # run once to embed your images (CLIP)
    ├── grader.py             # relevance grading logic
    ├── rewriter.py           # query rewriting logic
    ├── web_search.py         # Tavily web search fallback
    └── pipeline.py           # ties everything together
```

## Setup (step by step)

### 1. Install Python dependencies

```bash
cd agentic-rag
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This will take a few minutes the first time — it downloads PyTorch and
transformers libraries.

### 2. Get your free API keys

You need two free API keys. Neither requires a credit card.

**Gemini API key (for the LLM):**
1. Go to https://aistudio.google.com/app/apikey
2. Sign in with a Google account
3. Click "Create API key"
4. Copy the key

**Tavily API key (for the web search fallback):**
1. Go to https://app.tavily.com
2. Sign up for a free account
3. Your API key is shown on your dashboard immediately (free tier includes
   ~1000 searches/month)

### 3. Add your keys

```bash
cp .env.example .env
```

Open `.env` and paste your keys in:

```
GEMINI_API_KEY=paste_your_key_here
TAVILY_API_KEY=paste_your_key_here
```

### 4. Add some documents

Drop a few PDFs or `.txt` files into `data/docs/`. For a portfolio demo,
good options are: a company handbook, a research paper, product
documentation, or lecture notes — anything with enough content to ask
multiple questions about.

(Optional) Drop a few images into `data/images/` if you want to test the
CLIP-based image retrieval path.

### 5. Ingest your documents

```bash
python src/ingest_text.py
python src/ingest_images.py     # optional, only if you added images
```

This embeds everything and stores it in a local `chroma_db/` folder — this
only needs to be re-run when you add new documents.

### 6. Run the app

```bash
streamlit run app.py
```

This opens a browser tab where you can ask questions and watch the
reasoning trace (retrieval → grading → correction → answer) in real time.

## What to say about this project on your resume/interviews

- Built a **self-correcting RAG pipeline** that grades its own retrieval
  quality before answering, instead of blindly trusting the top-k results
  — addressing a well-known failure mode of naive RAG systems.
- Implemented **query rewriting** and a **live web search fallback** as
  automatic recovery paths when local document retrieval fails.
- Integrated a **pretrained CLIP model** for image-based retrieval,
  demonstrating multimodal embedding usage without needing to train a
  model from scratch.
- Stack: Python, ChromaDB, Sentence-Transformers, Gemini API, Tavily API,
  Streamlit.

## Honest limitations to mention if asked

- The relevance grader uses an LLM call, so it adds latency compared to
  simple top-k retrieval — a real production system might use a cheaper
  classifier for grading instead.
- CLIP retrieval here is embedding-based similarity search, not object
  detection/classification — it's a lighter form of computer vision than
  training or fine-tuning a model.
- No evaluation dataset is included — for a stronger project, consider
  writing 10-15 sample Q&A pairs and measuring accuracy before/after
  adding the self-correction logic, to quantify the improvement.
