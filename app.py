"""Streamlit interface for the AI Research Assistant."""
import hashlib
import os

import streamlit as st
from dotenv import load_dotenv

from rag import Retriever, answer_question, extract_pdf, split_pages

load_dotenv()
st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide")
st.title("📚 AI Research Assistant")
st.caption("Upload text-based PDFs, retrieve evidence, and ask grounded questions with citations.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API key (optional for retrieval)", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    top_k = st.slider("Retrieved passages", 1, 8, 4)
    st.info("Documents are processed in memory. Retrieved excerpts are sent to OpenAI only when you generate an answer with an API key.")

uploads = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
if not uploads:
    st.info("Upload one or more PDFs to begin.")
    st.stop()

signature = tuple((f.name, hashlib.sha256(f.getvalue()).hexdigest()) for f in uploads)
if st.session_state.get("signature") != signature:
    pages = []
    errors = []
    for uploaded in uploads:
        try:
            pages.extend(extract_pdf(uploaded.getvalue(), uploaded.name))
        except Exception as exc:
            errors.append(f"{uploaded.name}: {exc}")
    st.session_state["signature"] = signature
    st.session_state["chunks"] = split_pages(pages)
    st.session_state["errors"] = errors

for error in st.session_state["errors"]:
    st.warning(f"Could not read {error}")
chunks = st.session_state["chunks"]
if not chunks:
    st.error("No selectable PDF text found. Scanned PDFs require OCR, which this version does not include.")
    st.stop()

retriever = Retriever(chunks)
st.success(f"Indexed {len(uploads)} PDF(s) into {len(chunks)} searchable passages.")
question = st.text_input("Ask a question about your documents")
if st.button("Find answer", type="primary", disabled=not question.strip()):
    hits = retriever.search(question, k=top_k)
    if not hits:
        st.warning("No relevant excerpts found. Try rephrasing your question.")
    else:
        with st.spinner("Preparing answer..."):
            try:
                response = answer_question(question, hits, api_key, model)
                st.subheader("Answer")
                st.write(response)
            except Exception as exc:
                st.error(f"Answer generation failed: {exc}")
        st.subheader("Source passages")
        for i, (chunk, score) in enumerate(hits, 1):
            with st.expander(f"[{i}] {chunk.source} — page {chunk.page} (similarity {score:.2f})", expanded=i == 1):
                st.write(chunk.text)
