from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

from rag_chatbot.config import RERANK_MODEL


@lru_cache(maxsize=1)
def _get_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError("sentence-transformers is required for reranking.") from e
    return CrossEncoder(RERANK_MODEL)


def rerank(query: str, docs: list[Document], top_k: int) -> list[Document]:
    if not docs:
        return docs
    try:
        ce = _get_cross_encoder()
        pairs = [(query, d.page_content) for d in docs]
        scores = ce.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[:top_k]]
    except Exception:
        return docs[:top_k]
