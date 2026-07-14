"""
IntelliRAG - Agentic Multimodal RAG

Run:
streamlit run app.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from pipeline import run_pipeline

st.set_page_config(
    page_title="IntelliRAG",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 IntelliRAG")
st.subheader("Agentic Multimodal Retrieval-Augmented Generation")

st.markdown(
"""
Ask questions about your uploaded PDFs.

**Features**
- 📄 Semantic document search
- 🌐 Web Search fallback (Tavily)
- 🤖 Gemini-powered answer generation
- 📚 ChromaDB vector database
- ⚡ Fast retrieval pipeline
"""
)

question = st.text_input(
    "Ask a question",
    placeholder="Example: What skills are mentioned in Diksha's resume?"
)

if st.button("🚀 Ask", use_container_width=True):

    if question.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    progress = st.empty()

    logs = []

    def logger(message):
        logs.append(message)
        progress.info("\n".join(logs))

    with st.spinner("Searching documents..."):

        result = run_pipeline(
            question,
            log=logger
        )

    progress.empty()

    st.success("Answer Generated")

    st.markdown("## 💬 Answer")

    st.write(result["answer"])

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Source", result["source"])

    with col2:
        st.metric("Retrieved Steps", len(result["trace"]))

    with st.expander("🔍 Reasoning Trace"):

        for item in result["trace"]:
            st.write("•", item)