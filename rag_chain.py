"""
rag_chain.py  —  RAG retrieval chain
---------------------------------------
Pipeline:
  retrieve (vector | hybrid)  →  optional rerank  →  build prompt
  (with optional chat history)  →  Llama 3 via Ollama  →  cited answer

Two query entrypoints:
  • query(...)         — blocking, returns full RAGResult
  • stream_query(...)  — generator yielding (token, sources) for streaming UIs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    LLM_MODEL,
    MAX_HISTORY_TURNS,
    MMR_FETCH_K,
    OLLAMA_BASE_URL,
    RERANK_FETCH_K,
    TOP_K,
    USE_HYBRID,
    USE_MMR,
    USE_RERANK,
)


# ── Prompt templates ───────────────────────────────────────────────────────

BASE_RULES = """You are a precise document assistant. Answer the user's question using ONLY the context passages provided below.

Rules:
1. Base your answer solely on the context. Do NOT use prior knowledge.
2. After each factual claim, add a citation in the format: [Source: <filename>, Page <number>]
3. If the answer is not found in the context, respond exactly: "I don't know based on the provided documents."
4. Be concise and factual. Do not repeat the question."""

# Single-turn prompt (no history)
RAG_PROMPT_TEMPLATE = BASE_RULES + """

Context passages:
{context}

Question: {question}

Answer (with citations):"""

# Multi-turn prompt (with chat history)
RAG_PROMPT_TEMPLATE_WITH_HISTORY = BASE_RULES + """
5. Use the conversation history only to understand what the user is referring to (e.g. "that paper", "the second point"). Do not invent facts from it.

Conversation history:
{history}

Context passages:
{context}

Current question: {question}

Answer (with citations):"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE,
)

RAG_PROMPT_WITH_HISTORY = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=RAG_PROMPT_TEMPLATE_WITH_HISTORY,
)


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class RAGResult:
    answer:  str
    sources: list[dict] = field(default_factory=list)
    # sources: list of {"source": str, "page": int, "text": str}


# ── Internal helpers ───────────────────────────────────────────────────────

def _get_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def _format_context(docs: list[Document]) -> str:
    """Render retrieved chunks into the {context} block, tagged with citations."""
    blocks = []
    for d in docs:
        src  = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        blocks.append(f"[Source: {src}, Page {page}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def _format_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> str:
    """
    Render the last `max_turns` user/assistant pairs as plain text.
    `history` is a list of {"role": "user"|"assistant", "content": str}.
    """
    if not history:
        return "(no prior turns)"
    # Keep last 2 * max_turns messages (one pair = 2 messages)
    recent = history[-(2 * max_turns):]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _docs_to_sources(docs: list[Document]) -> list[dict]:
    """Dedup retrieved docs by (source, page) for UI display."""
    sources, seen = [], set()
    for doc in docs:
        meta = doc.metadata
        key = (meta.get("source", ""), meta.get("page", 0))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "source": meta.get("source", "unknown"),
            "page":   meta.get("page", "?"),
            "text":   doc.page_content[:300],
        })
    return sources


def _retrieve(
    question: str,
    top_k: int,
    use_mmr: bool,
    use_hybrid: bool,
    use_rerank: bool,
) -> list[Document]:
    """
    Run the configured retrieval pipeline and return the final ordered docs.
    Reranking, if enabled, oversamples then reranks down to top_k.
    """
    vs = _get_vectorstore()
    fetch_k = RERANK_FETCH_K if use_rerank else top_k

    # Hybrid path
    if use_hybrid:
        from ingest import get_all_chunks
        from hybrid_retriever import build_hybrid_retriever
        try:
            hybrid = build_hybrid_retriever(vs, get_all_chunks(), top_k=fetch_k)
            docs = hybrid.invoke(question)
        except Exception:
            # Fall through to vector retrieval if hybrid setup fails
            docs = _vector_retrieve(vs, question, fetch_k, use_mmr)
    else:
        docs = _vector_retrieve(vs, question, fetch_k, use_mmr)

    # Optional rerank
    if use_rerank and docs:
        from reranker import rerank
        docs = rerank(question, docs, top_k=top_k)
    else:
        docs = docs[:top_k]

    return docs


def _vector_retrieve(vs: Chroma, question: str, k: int, use_mmr: bool) -> list[Document]:
    if use_mmr:
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": MMR_FETCH_K},
        )
    else:
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever.invoke(question)


# ── Public API: blocking query ─────────────────────────────────────────────

def query(
    question:    str,
    model:       str  = LLM_MODEL,
    top_k:       int  = TOP_K,
    use_mmr:     bool = USE_MMR,
    use_hybrid:  bool = USE_HYBRID,
    use_rerank:  bool = USE_RERANK,
    temperature: float = 0.1,
    history:     Optional[list[dict]] = None,
) -> RAGResult:
    """Run a question through the RAG pipeline and return the full answer + sources."""
    docs = _retrieve(question, top_k, use_mmr, use_hybrid, use_rerank)
    context = _format_context(docs)

    llm = OllamaLLM(model=model, base_url=OLLAMA_BASE_URL, temperature=temperature)

    if history:
        prompt = RAG_PROMPT_WITH_HISTORY.format(
            history=_format_history(history),
            context=context,
            question=question,
        )
    else:
        prompt = RAG_PROMPT.format(context=context, question=question)

    answer = llm.invoke(prompt).strip()
    return RAGResult(answer=answer, sources=_docs_to_sources(docs))


# ── Public API: streaming query ────────────────────────────────────────────

def stream_query(
    question:    str,
    model:       str  = LLM_MODEL,
    top_k:       int  = TOP_K,
    use_mmr:     bool = USE_MMR,
    use_hybrid:  bool = USE_HYBRID,
    use_rerank:  bool = USE_RERANK,
    temperature: float = 0.1,
    history:     Optional[list[dict]] = None,
) -> Generator[dict, None, None]:
    """
    Yield events as the answer is generated. Event shapes:
      {"type": "sources", "sources": [...]}     — emitted once, before generation
      {"type": "token",   "token":   "..."}     — emitted per LLM token
      {"type": "done"}                          — emitted once at the end

    The caller is responsible for accumulating tokens into the final answer string.
    """
    docs = _retrieve(question, top_k, use_mmr, use_hybrid, use_rerank)
    sources = _docs_to_sources(docs)
    yield {"type": "sources", "sources": sources}

    context = _format_context(docs)
    llm = OllamaLLM(model=model, base_url=OLLAMA_BASE_URL, temperature=temperature)

    if history:
        prompt = RAG_PROMPT_WITH_HISTORY.format(
            history=_format_history(history),
            context=context,
            question=question,
        )
    else:
        prompt = RAG_PROMPT.format(context=context, question=question)

    for chunk in llm.stream(prompt):
        if chunk:
            yield {"type": "token", "token": chunk}

    yield {"type": "done"}


# ── Hallucination guard ────────────────────────────────────────────────────

def validate_citations(result: RAGResult, ingested_pages: dict[str, int]) -> list[str]:
    """
    Check that cited page numbers actually exist in ingested documents.
    Returns list of warning strings (empty = all citations valid).
    """
    import re
    warnings = []
    pattern = re.compile(r"\[Source:\s*(.+?),\s*Page\s*(\d+)\]", re.IGNORECASE)
    for match in pattern.finditer(result.answer):
        fname, page_str = match.group(1).strip(), match.group(2)
        page = int(page_str)
        if fname in ingested_pages:
            if page > ingested_pages[fname]:
                warnings.append(
                    f"⚠️ Citation error: '{fname}' only has "
                    f"{ingested_pages[fname]} pages, but page {page} was cited."
                )
        else:
            warnings.append(f"⚠️ Citation error: '{fname}' is not in the ingested documents.")
    return warnings


# ── CLI convenience ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Summarise the main topics."
    print(f"\nQuestion: {q}\n")
    res = query(q)
    print(f"Answer:\n{res.answer}\n")
    print("Sources:")
    for s in res.sources:
        print(f"  • {s['source']}  page {s['page']}")
