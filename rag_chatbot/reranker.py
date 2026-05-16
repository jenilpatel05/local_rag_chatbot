"""
reranker.py  —  Cross-encoder re-ranking
-----------------------------------------
After the initial retriever pulls top-N candidates, a cross-encoder scores
each (query, chunk) pair jointly and we keep the top-K most relevant.

Cross-encoders are slower than bi-encoders (the embedding model that does
initial retrieval) but much more accurate — they look at the query and
the chunk together rather than embedding each independently.

The model is loaded once and cached for the lifetime of the process.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

from rag_chatbot.config import RERANK_MODEL


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Load the cross-encoder once (downloads ~80MB on first use)."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for reranking. "
            "Install with: pip install sentence-transformers"
        ) from e
    return CrossEncoder(RERANK_MODEL)


def rerank(query: str, docs: list[Document], top_k: int) -> list[Document]:
    """
    Re-score `docs` against `query` using the cross-encoder and return the top_k.
    If reranking fails for any reason, fall back to the original ordering.
    """
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
