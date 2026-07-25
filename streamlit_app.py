"""
streamlit_app.py
------------------
Chat UI for the PL-300 RAG tutor. Requires vector_db/ to already be built
(run 03_chunking.py then 05_create_index_store.py first).
"""

import streamlit as st
from importlib import import_module

prompting_mod = import_module("07_prompting")

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
        margin-bottom: 1.75rem;
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
    }
    .sidebar-card h4 {
        margin: 0 0 0.4rem 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

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
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=5,
                  help="How many context chunks the model sees before answering.")
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

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------------------------------
# Chat state
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

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

# ----------------------------------------------------------------------------
# Render chat history
# ----------------------------------------------------------------------------
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "📊"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("chunks"):
            with st.expander("📚 Retrieved context"):
                for i, chunk in enumerate(msg["chunks"], 1):
                    st.markdown(f"**Chunk {i}**")
                    st.text(chunk[:800])

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
        with st.spinner("Searching the knowledge base and thinking..."):
            try:
                answer, chunks = prompting_mod.answer_question(question, k=k)
            except Exception as e:
                answer = (
                    "⚠️ Something went wrong while generating the answer.\n\n"
                    f"`{e}`\n\nCheck that `OPENROUTER_API_KEY` is set correctly in your `.env` file."
                )
                chunks = []
        st.markdown(answer)
        if show_sources and chunks:
            with st.expander("📚 Retrieved context"):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.text(chunk[:800])

    st.session_state.messages.append({"role": "assistant", "content": answer, "chunks": chunks})
