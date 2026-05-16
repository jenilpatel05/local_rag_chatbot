from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

from rag_chatbot.config import HYBRID_FETCH_K, HYBRID_VECTOR_WEIGHT


class HybridRetriever(BaseRetriever):
    vector_retriever: object = Field(...)
    bm25_retriever: object = Field(...)
    top_k: int = Field(default=5)
    vector_weight: float = Field(default=HYBRID_VECTOR_WEIGHT)
    fetch_k: int = Field(default=HYBRID_FETCH_K)
    sources_filter: Optional[list[str]] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def _vector_filter(self) -> Optional[dict]:
        if not self.sources_filter:
            return None
        if len(self.sources_filter) == 1:
            return {"source": self.sources_filter[0]}
        return {"source": {"$in": list(self.sources_filter)}}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        try:
            self.bm25_retriever.k = self.fetch_k
        except Exception:
            pass
        bm25_docs = self.bm25_retriever.invoke(query)

        vs = self.vector_retriever
        vec_kwargs = {"k": self.fetch_k}
        flt = self._vector_filter()
        if flt is not None:
            vec_kwargs["filter"] = flt
        vec_results = vs.similarity_search_with_score(query, **vec_kwargs)

        def key(doc: Document) -> str:
            return doc.metadata.get("chunk_id") or str(hash(doc.page_content))

        vec_sims = {key(doc): 1.0 / (1.0 + float(dist)) for doc, dist in vec_results}
        vec_docs = {key(doc): doc for doc, _ in vec_results}

        n = max(len(bm25_docs), 1)
        bm25_scores = {key(doc): 1.0 - (i / n) for i, doc in enumerate(bm25_docs)}
        bm25_docs_map = {key(doc): doc for doc in bm25_docs}

        if vec_sims:
            lo, hi = min(vec_sims.values()), max(vec_sims.values())
            span = hi - lo or 1.0
            vec_norm = {k: (v - lo) / span for k, v in vec_sims.items()}
        else:
            vec_norm = {}

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


def build_hybrid_retriever(
    vectorstore,
    all_chunks: list[dict],
    top_k: int,
    sources_filter: Optional[list[str]] = None,
):
    try:
        from langchain_community.retrievers import BM25Retriever
    except ImportError as e:
        raise ImportError("rank_bm25 is required for hybrid search.") from e

    if not all_chunks:
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
        sources_filter=sources_filter,
    )
