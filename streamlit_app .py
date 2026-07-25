"""
streamlit_app.py
------------------
Chat UI for the PL-300 RAG tutor. Requires vector_db/ to already be built
(run 03_chunking.py then 05_create_index_store.py first).
"""

import os
import time
import pandas as pd
import streamlit as st
from importlib import import_module

prompting_mod = import_module("07_prompting")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="PL-300 RAG Tutor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 780px;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 1.75rem 1.5rem 1.5rem 1.5rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #2F5CFF 0%, #7C3AED 100%);
        color: white;
        margin-bottom: 1.25rem;
    }
    .hero h1 {
        font-size: 1.7rem;
        margin: 0 0 0.35rem 0;
        color: white;
    }
    .hero p {
        margin: 0;
        opacity: 0.92;
        font-size: 0.95rem;
    }
    .hero .badges {
        margin-top: 0.9rem;
        display: flex;
        gap: 0.5rem;
        justify-content: center;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
    }

    /* Suggested question chips */
    div[data-testid="stButton"] > button {
        border-radius: 10px;
        border: 1px solid #E2E5F1;
        background: #F4F6FB;
        color: #1A1D29;
        font-size: 0.85rem;
        padding: 0.55rem 0.8rem;
        text-align: left;
        white-space: normal;
        line-height: 1.25;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #2F5CFF;
        color: #2F5CFF;
        background: #EDF1FF;
    }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.35rem 0.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #F9FAFC;
    }
    .sidebar-card {
        background: white;
        border: 1px solid #E9EBF3;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        line-height: 1.5;
    }
    .sidebar-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.85rem;
    }

    /* Context blocks */
    .context-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #4A4F63;
        margin: 0.6rem 0 0.2rem 0;
        border-top: 1px dashed #E2E5F1;
        padding-top: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Helpers: system stats (best-effort, won't crash if files aren't built yet)
# ----------------------------------------------------------------------------
def get_system_stats():
    doc_count = "—"
    chunk_count = "—"
    try:
        pdf_count = 0
        for sub in ("microsoft_learn", "dumps"):
            folder = os.path.join(BASE_DIR, "data", sub)
            if os.path.isdir(folder):
                pdf_count += len([f for f in os.listdir(folder) if f.endswith(".pdf")])
        if pdf_count:
            doc_count = str(pdf_count)
    except Exception:
        pass

    try:
        chunks_path = os.path.join(BASE_DIR, "processed_data", "chunks.csv")
        if os.path.exists(chunks_path):
            chunk_count = str(len(pd.read_csv(chunks_path)))
    except Exception:
        pass

    return doc_count, chunk_count


DOC_COUNT, CHUNK_COUNT = get_system_stats()
MODEL_NAME = os.environ.get("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>📊 PL-300 RAG Tutor</h1>
    <p>Ask anything about the Power BI Data Analyst exam — answers are grounded in your Microsoft Learn docs and exam material, not guesses.</p>
    <div class="badges">
        <span class="badge">🔎 Retrieval-Augmented</span>
        <span class="badge">📚 Source-grounded</span>
        <span class="badge">⚡ OpenRouter-powered</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Chat state (init early so the metrics/sidebar below can read it)
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------------------------------------------------
# Top metrics
# ----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("📄 Documents", DOC_COUNT)
col2.metric("✂️ Chunks", CHUNK_COUNT)
col3.metric("🔍 Top-K", st.session_state.get("k", 5))
col4.metric("🤖 Model", MODEL_NAME.split("/")[-1])

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=5,
                  help="How many context chunks the model sees before answering.")
    st.session_state["k"] = k
    show_sources = st.toggle("Show retrieved context", value=False)

    st.markdown("---")
    st.markdown("""
    <div class="sidebar-card">
        <h4>ℹ️ About</h4>
        This tutor retrieves the most relevant passages from your indexed
        PDFs, then asks an LLM to answer <b>using only that context</b> —
        reducing hallucinations on exam-specific details.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-card">
        <h4>📈 System Info</h4>
        <b>Embedding</b><br>all-MiniLM-L6-v2
        <br><br>
        <b>Vector Database</b><br>FAISS IndexFlatL2
        <br><br>
        <b>LLM</b><br>OpenRouter ({MODEL_NAME})
        <br><br>
        <b>Knowledge Base</b><br>Microsoft Learn + PL-300
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"💬 Messages: {len(st.session_state.messages)}")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

SUGGESTED_QUESTIONS = [
    "What is DAX?",
    "What's the difference between measures and calculated columns?",
    "What is filter context in DAX?",
    "How do I configure row-level security?",
]

# Empty state: welcome + suggested questions
if not st.session_state.messages:
    st.markdown("#### 👋 Where should we start?")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"suggested_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()


def render_sources(chunks):
    with st.expander("📚 Retrieved context"):
        for i, chunk in enumerate(chunks, 1):
            st.markdown(f'<div class="context-label">📄 Context {i}</div>', unsafe_allow_html=True)
            st.code(chunk[:800], language="text")


# ----------------------------------------------------------------------------
# Render chat history
# ----------------------------------------------------------------------------
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "📊"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant":
            with st.container(border=True):
                st.markdown(msg["content"])
            if "elapsed" in msg:
                st.caption(f"⏱ Response time: {msg['elapsed']:.2f} sec")
        else:
            st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("chunks"):
            render_sources(msg["chunks"])

# ----------------------------------------------------------------------------
# Handle input (typed or a suggested-question click)
# ----------------------------------------------------------------------------
question = st.chat_input("Ask a PL-300 question...")
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📊"):
        with st.spinner("🔎 Retrieving relevant documents..."):
            start = time.time()
            try:
                answer, chunks = prompting_mod.answer_question(question, k=k)
            except Exception as e:
                answer = (
                    "⚠️ Something went wrong while generating the answer.\n\n"
                    f"`{e}`\n\nCheck that `OPENROUTER_API_KEY` is set correctly in your `.env` file."
                )
                chunks = []
            elapsed = time.time() - start

        if chunks:
            st.toast("🧠 Answer generated!", icon="✅")

        with st.container(border=True):
            st.markdown(answer)
        st.caption(f"⏱ Response time: {elapsed:.2f} sec")

        if show_sources and chunks:
            render_sources(chunks)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": chunks,
        "elapsed": elapsed,
    })

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.divider()
st.markdown("""
<center>

Made with ❤️ using
**FAISS • SentenceTransformers • OpenRouter • Streamlit**

PL-300 RAG Tutor © 2026

</center>
""", unsafe_allow_html=True)
