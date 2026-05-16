from __future__ import annotations

from rag_chatbot.ingest import chunk_pages


def _page(text: str, page: int = 1, source: str = "doc.txt") -> dict:
    return {"text": text, "page": page, "source": source}


def test_chunk_pages_deterministic_chunk_ids():
    pages = [_page("Lorem ipsum dolor sit amet. " * 50)]
    first = chunk_pages(pages, chunk_size=200, chunk_overlap=20)
    second = chunk_pages(pages, chunk_size=200, chunk_overlap=20)
    assert [c["metadata"]["chunk_id"] for c in first] == [c["metadata"]["chunk_id"] for c in second]


def test_chunk_pages_respects_chunk_size():
    pages = [_page("A sentence. " * 200)]
    small = chunk_pages(pages, chunk_size=100, chunk_overlap=10)
    large = chunk_pages(pages, chunk_size=500, chunk_overlap=10)
    assert len(small) > len(large)


def test_chunk_metadata_preserves_source_and_page():
    pages = [_page("hello world", page=3, source="myfile.pdf")]
    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=0)
    assert chunks
    md = chunks[0]["metadata"]
    assert md["source"] == "myfile.pdf"
    assert md["page"] == 3
    assert "chunk_id" in md and md["chunk_id"]


def test_chunk_ids_unique_within_run():
    pages = [
        _page("alpha " * 100, page=1, source="a.txt"),
        _page("beta " * 100, page=2, source="a.txt"),
    ]
    chunks = chunk_pages(pages, chunk_size=150, chunk_overlap=10)
    ids = [c["metadata"]["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_empty_pages_returns_empty():
    assert chunk_pages([]) == []
