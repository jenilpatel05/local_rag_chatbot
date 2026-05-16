from __future__ import annotations

from rag_chatbot.rag_chain import _build_source_filter


def test_none_returns_none():
    assert _build_source_filter(None) is None


def test_empty_list_returns_none():
    assert _build_source_filter([]) is None


def test_single_source_uses_equality():
    assert _build_source_filter(["a.pdf"]) == {"source": "a.pdf"}


def test_multiple_sources_use_in_operator():
    out = _build_source_filter(["a.pdf", "b.txt", "c.md"])
    assert out == {"source": {"$in": ["a.pdf", "b.txt", "c.md"]}}


def test_input_is_not_mutated():
    sources = ["a.pdf", "b.pdf"]
    out = _build_source_filter(sources)
    sources.append("c.pdf")
    assert out == {"source": {"$in": ["a.pdf", "b.pdf"]}}
