"""
Efficient Agentic RAG Pipeline

Flow:
1. Retrieve top-k documents from ChromaDB
2. Send ALL retrieved context to Gemini
3. If Gemini says information is insufficient -> use Tavily Web Search
4. Generate final answer
"""

import os
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from web_search import web_search

load_dotenv()

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

TOP_K = 10

ANSWER_PROMPT = """
You are an AI assistant.

Answer the user's question ONLY using the retrieved context.

If the answer exists in the context:
- answer confidently
- summarize where appropriate
- mention which document(s) the information came from

If the answer does NOT exist,
reply exactly:

INSUFFICIENT_CONTEXT

Context
------------------
{context}
------------------

Question:
{question}
"""


def get_gemini():
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    return genai.GenerativeModel("gemini-2.5-flash")


def retrieve_text(embed_model, collection, question, k=TOP_K):

    query_embedding = embed_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    docs = results.get("documents", [[]])[0]

    return docs


def run_pipeline(question, image_path=None, log=None):

    def _log(msg):
        if log:
            log(msg)

    gemini = get_gemini()

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=DB_DIR)

    text_collection = client.get_or_create_collection("text_chunks")

    trace = []

    _log("Retrieving relevant documents...")

    chunks = retrieve_text(
        embed_model,
        text_collection,
        question,
    )

    trace.append(f"Retrieved {len(chunks)} chunks")

    if not chunks:

        return {
            "answer": "No local documents were found.",
            "source": "none",
            "trace": trace,
        }

    context = "\n\n".join(chunks)

    _log("Generating answer from local documents...")

    answer = gemini.generate_content(
        ANSWER_PROMPT.format(
            context=context,
            question=question,
        )
    ).text

    if "INSUFFICIENT_CONTEXT" not in answer:

        return {
            "answer": answer,
            "source": "Local Documents",
            "trace": trace,
        }

    _log("Falling back to Tavily Web Search...")

    snippets = web_search(
        os.environ["TAVILY_API_KEY"],
        question,
    )

    trace.append(f"Web search returned {len(snippets)} snippets")

    if not snippets:

        return {
            "answer": "I couldn't find enough information locally or on the web.",
            "source": "None",
            "trace": trace,
        }

    context = "\n\n".join(snippets)

    FINAL_PROMPT = f"""
Use the web search context below to answer.

{context}

Question:
{question}
"""

    answer = gemini.generate_content(FINAL_PROMPT).text

    return {
        "answer": answer,
        "source": "Web Search",
        "trace": trace,
    }