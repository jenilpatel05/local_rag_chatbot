"""
hybrid_retriever.py  —  BM25 + dense vector hybrid retrieval
-------------------------------------------------------------
Fuses keyword search (BM25) with semantic search (Chroma vectors) using
normalised score combination. Use when you want both exact-keyword recall
(names, codes, acronyms) AND semantic recall (paraphrases, concepts).
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

from config import HYBRID_FETCH_K, HYBRID_VECTOR_WEIGHT


class HybridRetriever(BaseRetriever):
    """
    Combines a BM25 retriever and a vector retriever, normalises both
    score sets to [0, 1] independently, then fuses with a weighted sum.

    Final score = w * vector_norm + (1 - w) * bm25_norm
    """

    vector_retriever: object = Field(...)
    bm25_retriever:   object = Field(...)
    top_k:            int   = Field(default=5)
    vector_weight:    float = Field(default=HYBRID_VECTOR_WEIGHT)
    fetch_k:          int   = Field(default=HYBRID_FETCH_K)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        # Pull a wide candidate pool from each retriever
        try:
            self.bm25_retriever.k = self.fetch_k
        except Exception:
            pass
        bm25_docs = self.bm25_retriever.invoke(query)

        # Chroma similarity_search_with_score returns (doc, distance). Lower is better
        # for cosine distance, so convert to similarity = 1 / (1 + distance).
        vs = self.vector_retriever
        vec_results = vs.similarity_search_with_score(query, k=self.fetch_k)

        # Build dictionaries keyed by chunk_id (or text hash as a fallback)
        def key(doc: Document) -> str:
            return doc.metadata.get("chunk_id") or str(hash(doc.page_content))

        # Vector scores: convert distance → similarity, then min-max normalise
        vec_sims = {key(doc): 1.0 / (1.0 + float(dist)) for doc, dist in vec_results}
        vec_docs = {key(doc): doc for doc, _ in vec_results}

        # BM25 doesn't return scores directly via invoke(), so we rank by position.
        # Rank-based score: 1 - (rank / N) gives a smooth descending score in [0, 1].
        n = max(len(bm25_docs), 1)
        bm25_scores = {key(doc): 1.0 - (i / n) for i, doc in enumerate(bm25_docs)}
        bm25_docs_map = {key(doc): doc for doc in bm25_docs}

        # Min-max normalise vec_sims to [0, 1]
        if vec_sims:
            lo, hi = min(vec_sims.values()), max(vec_sims.values())
            span = hi - lo or 1.0
            vec_norm = {k: (v - lo) / span for k, v in vec_sims.items()}
        else:
            vec_norm = {}

        # Fuse
        all_keys = set(vec_norm) | set(bm25_scores)
        w = self.vector_weight
        fused: list[tuple[float, Document]] = []
        for k in all_keys:
            score = w * vec_norm.get(k, 0.0) + (1 - w) * bm25_scores.get(k, 0.0)
            doc = vec_docs.get(k) or bm25_docs_map.get(k)
            if doc is not None:
                fused.append((score, doc))

        fused.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in fused[: self.top_k]]


def build_hybrid_retriever(vectorstore, all_chunks: list[dict], top_k: int):
    """
    Factory: takes the Chroma vectorstore and the full chunk list
    (needed for BM25), returns a HybridRetriever.
    """
    try:
        from langchain_community.retrievers import BM25Retriever
    except ImportError as e:
        raise ImportError(
            "rank_bm25 is required for hybrid search. "
            "Install with: pip install rank_bm25"
        ) from e

    if not all_chunks:
        # No documents yet — caller should fall back to pure vector
        raise ValueError("No documents ingested; cannot build BM25 index.")

    docs = [
        Document(page_content=c["text"], metadata=c["metadata"])
        for c in all_chunks
    ]
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = HYBRID_FETCH_K

    return HybridRetriever(
        vector_retriever=vectorstore,
        bm25_retriever=bm25,
        top_k=top_k,
    )
