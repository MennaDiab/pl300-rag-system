"""
streamlit_app.py
------------------
Chat UI for the PL-300 RAG tutor. Requires vector_db/ to already be built
(run 03_chunking.py then 05_create_index_store.py first).
"""

import streamlit as st
from importlib import import_module

prompting_mod = import_module("07_prompting")

st.set_page_config(page_title="PL-300 RAG Tutor", page_icon="📊", layout="centered")

st.title("📊 PL-300 RAG Tutor")
st.caption("Ask any question about the PL-300 (Power BI) exam. Answers are grounded in the Microsoft Learn docs + exam material you indexed.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    k = st.slider("Number of retrieved chunks (k)", min_value=1, max_value=10, value=5)
    show_sources = st.checkbox("Show retrieved context", value=False)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a PL-300 question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, chunks = prompting_mod.answer_question(question, k=k)
            except Exception as e:
                answer = f"Something went wrong: {e}"
                chunks = []
            st.markdown(answer)
            if show_sources and chunks:
                with st.expander("Retrieved context"):
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(f"**Chunk {i}**")
                        st.text(chunk[:800])

    st.session_state.messages.append({"role": "assistant", "content": answer})
