"""
streamlit_app.py
------------------
Chat UI for the PL-300 AI Guide (RAG tutor).
Requires vector_db/ to already be built
(run 03_chunking.py then 05_create_index_store.py first).

NOTE: This file is UI-only. All retrieval / embedding / FAISS / OpenRouter /
prompting logic lives in 06_retrieve_context.py and 07_prompting.py and is
untouched here — this module only calls prompting_mod.answer_question().
"""

import os
import time
import pandas as pd
import streamlit as st
from importlib import import_module

prompting_mod = import_module("07_prompting")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ==============================================================================
# Page config
# ==============================================================================
st.set_page_config(
    page_title="PL-300 AI Guide",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Custom CSS — dark, Power BI-inspired theme
# ==============================================================================
st.markdown("""
<style>
    :root {
        --pbi-yellow: #F2C811;
        --pbi-accent: #FFD54F;
        --pbi-bg: #0F1117;
        --pbi-card: #1B1F2A;
        --pbi-border: #2A2F3A;
        --pbi-text: #FFFFFF;
        --pbi-text-secondary: #B5BAC8;
        --pbi-success: #00C853;
    }

    #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}

    html, body, [class*="st-"] {
        color: var(--pbi-text);
    }

    .stApp {
        background: radial-gradient(circle at 20% -10%, #171b26 0%, var(--pbi-bg) 55%);
    }

    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 2rem;
        max-width: 820px;
    }

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--pbi-bg); }
    ::-webkit-scrollbar-thumb { background: var(--pbi-border); border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--pbi-yellow); }

    /* ---------- Hero ---------- */
    .hero {
        text-align: center;
        padding: 2rem 1.75rem 1.75rem 1.75rem;
        border-radius: 20px;
        background: linear-gradient(160deg, #1B1F2A 0%, #14161F 100%);
        border: 1px solid var(--pbi-border);
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        margin-bottom: 1.5rem;
    }
    .hero .icon-row {
        font-size: 1.6rem;
        margin-bottom: 0.6rem;
        opacity: 0.9;
    }
    .hero h1 {
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        color: var(--pbi-text);
        letter-spacing: -0.02em;
    }
    .hero h1 .accent { color: var(--pbi-yellow); }
    .hero .tagline {
        font-size: 1.02rem;
        color: var(--pbi-text-secondary);
        margin: 0 0 1.1rem 0;
    }
    .hero .topics {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
        margin-top: 0.5rem;
    }
    .topic-chip {
        background: rgba(242, 200, 17, 0.08);
        border: 1px solid rgba(242, 200, 17, 0.35);
        color: var(--pbi-accent);
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
    }

    /* ---------- Metric cards ---------- */
    div[data-testid="stMetric"] {
        background: var(--pbi-card);
        border: 1px solid var(--pbi-border);
        border-radius: 14px;
        padding: 0.8rem 0.6rem;
        text-align: center;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--pbi-yellow);
        transform: translateY(-2px);
    }
    div[data-testid="stMetricLabel"] { color: var(--pbi-text-secondary) !important; }
    div[data-testid="stMetricValue"] { color: var(--pbi-yellow) !important; font-weight: 700; }

    /* ---------- Buttons (suggestion cards + sidebar) ---------- */
    div[data-testid="stButton"] > button {
        border-radius: 12px;
        border: 1px solid var(--pbi-border);
        background: var(--pbi-card);
        color: var(--pbi-text);
        font-size: 0.87rem;
        padding: 0.7rem 0.9rem;
        text-align: left;
        white-space: normal;
        line-height: 1.3;
        transition: all 0.18s ease;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
    }
    div[data-testid="stButton"] > button:hover {
        border-color: var(--pbi-yellow);
        color: var(--pbi-yellow);
        background: #21263355;
        transform: translateY(-1px);
    }
    div[data-testid="stButton"] > button:active { transform: translateY(0); }

    /* Primary-style buttons (Clear chat) */
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        text-align: center;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #0C0E14;
        border-right: 1px solid var(--pbi-border);
    }
    section[data-testid="stSidebar"] h3 {
        color: var(--pbi-yellow) !important;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .sidebar-card {
        background: var(--pbi-card);
        border: 1px solid var(--pbi-border);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        line-height: 1.6;
        color: var(--pbi-text-secondary);
    }
    .sidebar-card h4 {
        margin: 0 0 0.6rem 0;
        font-size: 0.85rem;
        color: var(--pbi-text);
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .sidebar-card b { color: var(--pbi-text); }
    .sidebar-card .row {
        display: flex;
        justify-content: space-between;
        padding: 0.25rem 0;
        border-bottom: 1px dashed var(--pbi-border);
    }
    .sidebar-card .row:last-child { border-bottom: none; }
    .sidebar-card .row span:last-child { color: var(--pbi-accent); font-weight: 600; }

    /* ---------- Chat ---------- */
    div[data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.3rem 0;
        border: none;
    }
    div[data-testid="stChatInput"] textarea {
        background: var(--pbi-card) !important;
        border: 1px solid var(--pbi-border) !important;
        color: var(--pbi-text) !important;
        border-radius: 14px !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: var(--pbi-yellow) !important;
    }

    .user-bubble {
        background: linear-gradient(135deg, #2A2F3A 0%, #21252F 100%);
        border: 1px solid var(--pbi-border);
        border-radius: 16px 16px 4px 16px;
        padding: 0.7rem 1rem;
        margin-left: auto;
        max-width: 85%;
        color: var(--pbi-text);
        box-shadow: 0 2px 12px rgba(0,0,0,0.2);
    }
    .user-wrap { display: flex; justify-content: flex-end; }

    .assistant-card {
        background: var(--pbi-card);
        border: 1px solid var(--pbi-border);
        border-left: 3px solid var(--pbi-yellow);
        border-radius: 4px 16px 16px 16px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }

    /* ---------- Expanders (retrieved context) ---------- */
    div[data-testid="stExpander"] {
        background: var(--pbi-card);
        border: 1px solid var(--pbi-border);
        border-radius: 12px;
    }
    div[data-testid="stExpander"] summary {
        color: var(--pbi-text-secondary);
        font-size: 0.85rem;
    }

    .context-card {
        background: #14161F;
        border: 1px solid var(--pbi-border);
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.6rem;
    }
    .context-card .context-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        color: var(--pbi-accent);
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .context-card .context-score {
        color: var(--pbi-text-secondary);
        font-weight: 400;
    }

    /* ---------- Status / toast ---------- */
    div[data-testid="stStatusWidget"] {
        background: var(--pbi-card);
        border: 1px solid var(--pbi-border);
        border-radius: 12px;
    }

    /* ---------- Misc ---------- */
    hr { border-color: var(--pbi-border) !important; }
    .footer-text {
        text-align: center;
        color: var(--pbi-text-secondary);
        font-size: 0.82rem;
        line-height: 1.8;
    }
    .footer-text b { color: var(--pbi-accent); }

    @media (max-width: 640px) {
        .hero h1 { font-size: 1.5rem; }
        .user-bubble, .assistant-card { max-width: 100%; }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Helpers
# ==============================================================================
def get_system_stats():
    doc_count, chunk_count = "—", "—"
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


def stream_text(text, delay=0.012):
    """Cosmetic word-by-word streaming for st.write_stream (UI-only, no backend change)."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


def render_sources(chunks):
    with st.expander("📚 Retrieved Context", expanded=False):
        for i, chunk in enumerate(chunks, 1):
            st.markdown(f"""
            <div class="context-card">
                <div class="context-header">
                    <span>📄 Chunk {i}</span>
                    <span class="context-score">similarity score n/a</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.code(chunk[:800], language="text")


DOC_COUNT, CHUNK_COUNT = get_system_stats()
MODEL_NAME = os.environ.get("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response_time" not in st.session_state:
    st.session_state.last_response_time = None

# ==============================================================================
# Hero
# ==============================================================================
st.markdown("""
<div class="hero">
    <div class="icon-row">📊 ⚡ 🧠</div>
    <h1>📊 PL-300 <span class="accent">AI Guide</span></h1>
    <p class="tagline">Master Microsoft Power BI with an AI-powered learning assistant.</p>
    <div class="topics">
        <span class="topic-chip">DAX</span>
        <span class="topic-chip">Power Query</span>
        <span class="topic-chip">Data Modeling</span>
        <span class="topic-chip">Power BI Service</span>
        <span class="topic-chip">Reports &amp; Dashboards</span>
        <span class="topic-chip">Row-Level Security</span>
        <span class="topic-chip">Performance Optimization</span>
        <span class="topic-chip">PL-300 Certification</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# Top metrics
# ==============================================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("📄 Documents", DOC_COUNT)
c2.metric("✂️ Chunks", CHUNK_COUNT)
c3.metric("🔍 Top-K", st.session_state.get("k", 5))
rt = st.session_state.last_response_time
c4.metric("⚡ Response Time", f"{rt:.2f}s" if rt else "—")

# ==============================================================================
# Sidebar
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    k = st.slider("Top-K (chunks retrieved)", min_value=1, max_value=10, value=5,
                  help="How many context chunks the model sees before answering.")
    st.session_state["k"] = k
    show_sources = st.toggle("Show Retrieved Context", value=False)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_response_time = None
        st.rerun()

    st.markdown("---")

    st.markdown("""
    <div class="sidebar-card">
        <h4>ℹ️ About</h4>
        This assistant retrieves the most relevant passages from your indexed
        PDFs, then asks an LLM to answer <b>using only that context</b> —
        reducing hallucinations on exam-specific details.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-card">
        <h4>🧩 System Information</h4>
        <div class="row"><span>Embedding Model</span><span>all-MiniLM-L6-v2</span></div>
        <div class="row"><span>Vector Store</span><span>FAISS IndexFlatL2</span></div>
        <div class="row"><span>LLM</span><span>OpenRouter</span></div>
        <div class="row"><span>Knowledge Base</span><span>MS Learn + PL-300</span></div>
    </div>
    """, unsafe_allow_html=True)

    rt_display = f"{st.session_state.last_response_time:.2f}s" if st.session_state.last_response_time else "—"
    st.markdown(f"""
    <div class="sidebar-card">
        <h4>📊 Application Stats</h4>
        <div class="row"><span>Current Model</span><span>{MODEL_NAME.split('/')[-1]}</span></div>
        <div class="row"><span>Response Time</span><span>{rt_display}</span></div>
        <div class="row"><span>Indexed Documents</span><span>{DOC_COUNT}</span></div>
        <div class="row"><span>Indexed Chunks</span><span>{CHUNK_COUNT}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"💬 Messages in this session: {len(st.session_state.messages)}")

# ==============================================================================
# Welcome screen (clickable suggestion cards)
# ==============================================================================
SUGGESTED_QUESTIONS = [
    "What is DAX?",
    "Explain Query Folding",
    "Difference between Measure and Calculated Column",
    "What is Star Schema?",
    "Explain CALCULATE()",
    "What is RLS?",
    "How does Row Context differ from Filter Context?",
]

if not st.session_state.messages:
    st.markdown("#### 👋 What would you like to learn today?")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"suggested_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

# ==============================================================================
# Render chat history
# ==============================================================================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(f'<div class="user-wrap"><div class="user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="📊"):
            st.markdown(f'<div class="assistant-card">{msg["content"]}</div>', unsafe_allow_html=True)
            if "elapsed" in msg:
                st.caption(f"⚡ Response time: {msg['elapsed']:.2f}s")
            if show_sources and msg.get("chunks"):
                render_sources(msg["chunks"])

# ==============================================================================
# Handle input (typed or a suggested-question click)
# ==============================================================================
question = st.chat_input("Ask about DAX, Power Query, data modeling, RLS, and more...")
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(f'<div class="user-wrap"><div class="user-bubble">{question}</div></div>', unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="📊"):
        status = st.status("🔎 Searching knowledge base...", expanded=False)
        start = time.time()
        try:
            status.update(label="📚 Retrieving relevant documents...", state="running")
            status.update(label="🧠 Generating grounded answer...", state="running")
            answer, chunks = prompting_mod.answer_question(question, k=k)
            status.update(label="✅ Response ready", state="complete")
        except Exception as e:
            answer = (
                "⚠️ Something went wrong while generating the answer.\n\n"
                f"`{e}`\n\nCheck that `OPENROUTER_API_KEY` is set correctly in your `.env` file."
            )
            chunks = []
            status.update(label="⚠️ Error while generating answer", state="error")
        elapsed = time.time() - start

        if chunks:
            st.toast("🧠 Answer generated!", icon="✅")

        placeholder = st.empty()
        with placeholder.container():
            st.markdown('<div class="assistant-card">', unsafe_allow_html=True)
            st.write_stream(stream_text(answer))
            st.markdown('</div>', unsafe_allow_html=True)

        st.caption(f"⚡ Response time: {elapsed:.2f}s")

        if show_sources and chunks:
            render_sources(chunks)

    st.session_state.last_response_time = elapsed
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": chunks,
        "elapsed": elapsed,
    })

# ==============================================================================
# Footer
# ==============================================================================
st.divider()
st.markdown("""
<div class="footer-text">
    Powered by<br>
    <b>FAISS • Sentence Transformers • OpenRouter • Streamlit</b><br><br>
    Built for Microsoft PL-300 Certification © 2026
</div>
""", unsafe_allow_html=True)
