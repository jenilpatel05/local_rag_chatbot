"""
app.py  —  Local RAG Chatbot  (Streamlit UI)
---------------------------------------------
Run:  streamlit run app.py
"""

from __future__ import annotations

import time

import streamlit as st

from rag_chatbot.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    LLM_MODEL,
    MAX_HISTORY_TURNS,
    SUPPORTED_EXTENSIONS,
    TOP_K,
    UPLOADS_DIR,
    USE_CHAT_MEMORY,
    USE_HYBRID,
    USE_RERANK,
    USE_STREAMING,
)
from rag_chatbot.ingest import (
    delete_source,
    ingest_source,
    list_ingested_sources,
)
from rag_chatbot.rag_chain import BASE_RULES, query as rag_query, stream_query

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Local RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.source-card {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-left: 4px solid #4CAF50;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 13px;
}
.source-header {
    font-weight: 600;
    color: #2e7d32;
    margin-bottom: 4px;
}
.source-snippet {
    color: #555;
    font-style: italic;
    line-height: 1.5;
}
.chat-info {
    font-size: 12px;
    color: #888;
    margin-top: 4px;
}
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    background: #e8f5e9;
    color: #2e7d32;
    margin-right: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content, sources?, elapsed?}]
if "ingested" not in st.session_state:
    st.session_state.ingested = list_ingested_sources()
if "doc_filter" not in st.session_state:
    st.session_state.doc_filter = []

_DEFAULTS = {
    "adv_temperature":   0.1,
    "adv_chunk_size":    CHUNK_SIZE,
    "adv_chunk_overlap": CHUNK_OVERLAP,
    "adv_system_prompt": BASE_RULES,
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 Local RAG Chatbot")
    st.caption("Powered by Llama 3 · ChromaDB · Ollama")
    st.divider()

    # ── File upload (PDF / DOCX / TXT / MD) ────────────────────────────────
    st.markdown("### Upload Documents")
    accepted_types = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]
    uploaded_files = st.file_uploader(
        f"Drop files here ({', '.join(sorted(accepted_types))})",
        type=accepted_types,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uploaded in uploaded_files:
            dest = UPLOADS_DIR / uploaded.name
            dest.write_bytes(uploaded.getbuffer())
            with st.spinner(f"Ingesting {uploaded.name} …"):
                try:
                    pages, chunks = ingest_source(
                        dest,
                        chunk_size=st.session_state.adv_chunk_size,
                        chunk_overlap=st.session_state.adv_chunk_overlap,
                    )
                except Exception as e:
                    st.error(f"❌ {uploaded.name}: {e}")
                    continue
            if chunks > 0:
                st.success(f"✅ {uploaded.name}  ({pages} pages, {chunks} chunks)")
            else:
                st.info(f"ℹ️ {uploaded.name} already ingested (0 new chunks)")
        st.session_state.ingested = list_ingested_sources()

    # ── URL ingestion ──────────────────────────────────────────────────────
    with st.expander("🔗 Add a web page by URL"):
        url_input = st.text_input("URL", placeholder="https://example.com/article", label_visibility="collapsed")
        if st.button("Ingest URL", use_container_width=True):
            if url_input.strip():
                with st.spinner(f"Fetching and ingesting {url_input} …"):
                    try:
                        pages, chunks = ingest_source(
                            url_input.strip(),
                            chunk_size=st.session_state.adv_chunk_size,
                            chunk_overlap=st.session_state.adv_chunk_overlap,
                        )
                        if chunks > 0:
                            st.success(f"✅ Ingested {chunks} chunks from URL")
                        else:
                            st.info("ℹ️ URL already ingested (0 new chunks)")
                        st.session_state.ingested = list_ingested_sources()
                    except Exception as e:
                        st.error(f"❌ Failed: {e}")
            else:
                st.warning("Please enter a URL.")

    st.divider()

    # ── Ingested documents list ────────────────────────────────────────────
    st.markdown("### Ingested Documents")
    ingested = list_ingested_sources()
    if ingested:
        for fname in ingested:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"<span class='badge'>DOC</span> {fname}", unsafe_allow_html=True)
            if col2.button("🗑️", key=f"del_{fname}", help=f"Remove {fname}"):
                n = delete_source(fname)
                st.toast(f"Removed {n} chunks from {fname}")
                st.session_state.ingested = list_ingested_sources()
                st.session_state.doc_filter = [
                    f for f in st.session_state.doc_filter if f != fname
                ]
                st.rerun()
    else:
        st.caption("No documents yet — upload a file or URL above.")

    if ingested:
        st.session_state.doc_filter = [
            f for f in st.session_state.doc_filter if f in ingested
        ]
        st.multiselect(
            "Filter retrieval to selected docs",
            options=ingested,
            key="doc_filter",
            help="Empty = search all documents. Pick one or more to scope queries.",
        )

    st.divider()

    # ── Settings ───────────────────────────────────────────────────────────
    st.markdown("### Settings")
    model_choice = st.selectbox(
        "LLM model",
        ["llama3", "mistral", "phi3", "llama3:8b", "gemma2"],
        index=0,
    )
    top_k = st.slider("Chunks retrieved (top-K)", min_value=1, max_value=10, value=TOP_K)
    use_mmr     = st.toggle("MMR retrieval (diversity)",       value=True)
    use_hybrid  = st.toggle("Hybrid search (BM25 + vector)",   value=USE_HYBRID)
    use_rerank  = st.toggle("Cross-encoder rerank",            value=USE_RERANK)
    use_memory  = st.toggle("Chat memory (multi-turn)",        value=USE_CHAT_MEMORY)
    use_stream  = st.toggle("Stream tokens",                   value=USE_STREAMING)

    with st.expander("Advanced settings"):
        st.slider(
            "Temperature", min_value=0.0, max_value=1.0, step=0.05,
            key="adv_temperature",
        )
        st.number_input(
            "Chunk size (new uploads only)", min_value=100, max_value=2000, step=50,
            key="adv_chunk_size",
        )
        st.number_input(
            "Chunk overlap (new uploads only)", min_value=0, max_value=500, step=10,
            key="adv_chunk_overlap",
        )
        st.text_area(
            "System prompt", height=200,
            key="adv_system_prompt",
        )
        if st.button("Reset to defaults", use_container_width=True):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Everything runs locally. No data leaves your machine.")


# ── Main panel ─────────────────────────────────────────────────────────────
st.markdown("## 💬 Ask your documents")

if not list_ingested_sources():
    st.info("👈 Upload a document or add a URL in the sidebar to get started.")
else:
    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📄 {len(msg['sources'])} source(s) used", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f"""<div class='source-card'>
                                <div class='source-header'>
                                    📄 {src['source']} &nbsp;·&nbsp; Page {src['page']}
                                </div>
                                <div class='source-snippet'>"{src['text']}…"</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                if msg.get("elapsed"):
                    st.markdown(
                        f"<div class='chat-info'>⏱ {msg['elapsed']:.1f}s · "
                        f"{model_choice} · top-{top_k} chunks</div>",
                        unsafe_allow_html=True,
                    )

    # ── Chat input ─────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about your documents…"):
        # Build history BEFORE appending the current prompt
        history = (
            [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            if use_memory else None
        )

        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # ── Generate answer ────────────────────────────────────────────────
        with st.chat_message("assistant"):
            t0 = time.time()

            if use_stream:
                # Streaming path: write tokens as they arrive
                answer_placeholder = st.empty()
                sources_placeholder = st.empty()
                accumulated = ""
                sources = []

                try:
                    for event in stream_query(
                        question=prompt,
                        model=model_choice,
                        top_k=top_k,
                        use_mmr=use_mmr,
                        use_hybrid=use_hybrid,
                        use_rerank=use_rerank,
                        history=history,
                        sources_filter=st.session_state.doc_filter or None,
                        system_prompt=st.session_state.adv_system_prompt,
                        temperature=st.session_state.adv_temperature,
                    ):
                        if event["type"] == "sources":
                            sources = event["sources"]
                        elif event["type"] == "token":
                            accumulated += event["token"]
                            answer_placeholder.markdown(accumulated + "▌")
                        elif event["type"] == "done":
                            answer_placeholder.markdown(accumulated)
                    elapsed = time.time() - t0
                    answer = accumulated
                except Exception as e:
                    answer = (
                        f"❌ Error: {e}\n\n"
                        f"Make sure Ollama is running (`ollama serve`) "
                        f"and the model is pulled (`ollama pull {model_choice}`)."
                    )
                    answer_placeholder.markdown(answer)
                    sources = []
                    elapsed = time.time() - t0

                if sources:
                    with sources_placeholder.container():
                        with st.expander(f"📄 {len(sources)} source(s) used", expanded=True):
                            for src in sources:
                                st.markdown(
                                    f"""<div class='source-card'>
                                        <div class='source-header'>
                                            📄 {src['source']} &nbsp;·&nbsp; Page {src['page']}
                                        </div>
                                        <div class='source-snippet'>"{src['text']}…"</div>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )
                        st.markdown(
                            f"<div class='chat-info'>⏱ {elapsed:.1f}s · {model_choice} · "
                            f"top-{top_k} · "
                            f"{'hybrid' if use_hybrid else 'vector'}"
                            f"{' · reranked' if use_rerank else ''}"
                            f"{' · memory' if use_memory else ''}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            else:
                # Non-streaming path
                with st.spinner("Searching documents and generating answer…"):
                    try:
                        result = rag_query(
                            question=prompt,
                            model=model_choice,
                            top_k=top_k,
                            use_mmr=use_mmr,
                            use_hybrid=use_hybrid,
                            use_rerank=use_rerank,
                            history=history,
                            sources_filter=st.session_state.doc_filter or None,
                            system_prompt=st.session_state.adv_system_prompt,
                            temperature=st.session_state.adv_temperature,
                        )
                        elapsed = time.time() - t0
                        answer  = result.answer
                        sources = result.sources
                    except Exception as e:
                        answer = (
                            f"❌ Error: {e}\n\n"
                            f"Make sure Ollama is running (`ollama serve`) "
                            f"and the model is pulled (`ollama pull {model_choice}`)."
                        )
                        sources = []
                        elapsed = time.time() - t0

                st.markdown(answer)

                if sources:
                    with st.expander(f"📄 {len(sources)} source(s) used", expanded=True):
                        for src in sources:
                            st.markdown(
                                f"""<div class='source-card'>
                                    <div class='source-header'>
                                        📄 {src['source']} &nbsp;·&nbsp; Page {src['page']}
                                    </div>
                                    <div class='source-snippet'>"{src['text']}…"</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                    st.markdown(
                        f"<div class='chat-info'>⏱ {elapsed:.1f}s · {model_choice} · "
                        f"top-{top_k} · "
                        f"{'hybrid' if use_hybrid else 'vector'}"
                        f"{' · reranked' if use_rerank else ''}"
                        f"{' · memory' if use_memory else ''}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # Save assistant turn to history
        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources,
            "elapsed": elapsed,
        })
