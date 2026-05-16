from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Generator, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from rag_chatbot.config import (
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


BASE_RULES = """You are a precise document assistant. Answer the user's question using ONLY the context passages provided below.

Rules:
1. Base your answer solely on the context. Do NOT use prior knowledge.
2. After each factual claim, add a citation in the format: [Source: <filename>, Page <number>]
3. If the answer is not found in the context, respond exactly: "I don't know based on the provided documents."
4. Be concise and factual. Do not repeat the question."""


@dataclass
class RAGResult:
    answer: str
    sources: list[dict] = field(default_factory=list)


def _get_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        blocks.append(f"[Source: {src}, Page {page}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def _format_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> str:
    if not history:
        return "(no prior turns)"
    recent = history[-(2 * max_turns):]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _build_prompt(
    question: str,
    context: str,
    history: Optional[list[dict]],
    system_prompt: Optional[str],
) -> str:
    rules = system_prompt.strip() if system_prompt and system_prompt.strip() else BASE_RULES
    if history:
        history_rule = (
            "\n5. Use the conversation history only to understand what the user is "
            "referring to (e.g. \"that paper\", \"the second point\"). "
            "Do not invent facts from it."
        )
        tmpl = (
            rules + history_rule +
            "\n\nConversation history:\n{history}"
            "\n\nContext passages:\n{context}"
            "\n\nCurrent question: {question}"
            "\n\nAnswer (with citations):"
        )
        return tmpl.format(
            history=_format_history(history),
            context=context,
            question=question,
        )
    tmpl = (
        rules +
        "\n\nContext passages:\n{context}"
        "\n\nQuestion: {question}"
        "\n\nAnswer (with citations):"
    )
    return tmpl.format(context=context, question=question)


_REWRITE_PROMPT = """Given the conversation history and a follow-up question, rewrite the follow-up question as a standalone question that can be understood without the history. If the question is already standalone, return it unchanged. Output ONLY the rewritten question with no preamble.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""


def _rewrite_question(llm: "OllamaLLM", question: str, history: list[dict]) -> str:
    try:
        rewritten = llm.invoke(
            _REWRITE_PROMPT.format(history=_format_history(history), question=question)
        ).strip()
        rewritten = rewritten.strip('"').strip("'").strip()
        return rewritten or question
    except Exception:
        return question


def _docs_to_sources(docs: list[Document]) -> list[dict]:
    sources, seen = [], set()
    for doc in docs:
        meta = doc.metadata
        key = (meta.get("source", ""), meta.get("page", 0))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", "?"),
            "text": doc.page_content[:300],
        })
    return sources


def _build_source_filter(sources: Optional[list[str]]) -> Optional[dict]:
    if not sources:
        return None
    if len(sources) == 1:
        return {"source": sources[0]}
    return {"source": {"$in": list(sources)}}


def _retrieve(
    question: str,
    top_k: int,
    use_mmr: bool,
    use_hybrid: bool,
    use_rerank: bool,
    sources_filter: Optional[list[str]] = None,
) -> list[Document]:
    vs = _get_vectorstore()
    fetch_k = RERANK_FETCH_K if use_rerank else top_k

    if use_hybrid:
        from rag_chatbot.ingest import get_all_chunks
        from rag_chatbot.hybrid_retriever import build_hybrid_retriever
        try:
            chunks = get_all_chunks()
            if sources_filter:
                allowed = set(sources_filter)
                chunks = [c for c in chunks if c["metadata"].get("source") in allowed]
            hybrid = build_hybrid_retriever(
                vs, chunks, top_k=fetch_k, sources_filter=sources_filter,
            )
            docs = hybrid.invoke(question)
        except Exception:
            docs = _vector_retrieve(vs, question, fetch_k, use_mmr, sources_filter)
    else:
        docs = _vector_retrieve(vs, question, fetch_k, use_mmr, sources_filter)

    if use_rerank and docs:
        from rag_chatbot.reranker import rerank
        docs = rerank(question, docs, top_k=top_k)
    else:
        docs = docs[:top_k]

    return docs


def _vector_retrieve(
    vs: Chroma,
    question: str,
    k: int,
    use_mmr: bool,
    sources_filter: Optional[list[str]] = None,
) -> list[Document]:
    search_kwargs: dict = {"k": k}
    flt = _build_source_filter(sources_filter)
    if flt is not None:
        search_kwargs["filter"] = flt
    if use_mmr:
        search_kwargs["fetch_k"] = MMR_FETCH_K
        retriever = vs.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    else:
        retriever = vs.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    return retriever.invoke(question)


def query(
    question: str,
    model: str = LLM_MODEL,
    top_k: int = TOP_K,
    use_mmr: bool = USE_MMR,
    use_hybrid: bool = USE_HYBRID,
    use_rerank: bool = USE_RERANK,
    temperature: float = 0.1,
    history: Optional[list[dict]] = None,
    sources_filter: Optional[list[str]] = None,
    system_prompt: Optional[str] = None,
    rewrite_query: bool = False,
) -> RAGResult:
    llm = OllamaLLM(model=model, base_url=OLLAMA_BASE_URL, temperature=temperature)

    retrieval_q = (
        _rewrite_question(llm, question, history)
        if (rewrite_query and history) else question
    )

    docs = _retrieve(retrieval_q, top_k, use_mmr, use_hybrid, use_rerank, sources_filter)
    context = _format_context(docs)

    prompt = _build_prompt(question, context, history, system_prompt)
    answer = llm.invoke(prompt).strip()
    return RAGResult(answer=answer, sources=_docs_to_sources(docs))


def stream_query(
    question: str,
    model: str = LLM_MODEL,
    top_k: int = TOP_K,
    use_mmr: bool = USE_MMR,
    use_hybrid: bool = USE_HYBRID,
    use_rerank: bool = USE_RERANK,
    temperature: float = 0.1,
    history: Optional[list[dict]] = None,
    sources_filter: Optional[list[str]] = None,
    system_prompt: Optional[str] = None,
    rewrite_query: bool = False,
) -> Generator[dict, None, None]:
    llm = OllamaLLM(model=model, base_url=OLLAMA_BASE_URL, temperature=temperature)

    retrieval_q = (
        _rewrite_question(llm, question, history)
        if (rewrite_query and history) else question
    )

    docs = _retrieve(retrieval_q, top_k, use_mmr, use_hybrid, use_rerank, sources_filter)
    sources = _docs_to_sources(docs)
    yield {"type": "sources", "sources": sources}

    context = _format_context(docs)
    prompt = _build_prompt(question, context, history, system_prompt)

    for chunk in llm.stream(prompt):
        if chunk:
            yield {"type": "token", "token": chunk}

    yield {"type": "done"}


_CITATION_RE = re.compile(r"\[Source:\s*(.+?),\s*Page\s*(\d+)\]", re.IGNORECASE)


def validate_citations(result: RAGResult, ingested_pages: dict[str, int]) -> list[str]:
    warnings = []
    for match in _CITATION_RE.finditer(result.answer):
        fname, page_str = match.group(1).strip(), match.group(2)
        page = int(page_str)
        if fname in ingested_pages:
            if page > ingested_pages[fname]:
                warnings.append(
                    f"Citation error: '{fname}' only has "
                    f"{ingested_pages[fname]} pages, but page {page} was cited."
                )
        else:
            warnings.append(f"Citation error: '{fname}' is not in the ingested documents.")
    return warnings


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Summarise the main topics."
    print(f"\nQuestion: {q}\n")
    res = query(q)
    print(f"Answer:\n{res.answer}\n")
    print("Sources:")
    for s in res.sources:
        print(f"  - {s['source']}  page {s['page']}")
