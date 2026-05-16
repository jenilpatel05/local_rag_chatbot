from __future__ import annotations

from rag_chatbot.exporters import conversation_as_markdown


def test_empty_conversation_has_header_only():
    out = conversation_as_markdown([])
    lines = out.splitlines()
    assert lines[0].startswith("# Conversation export"), out
    assert "## **You**" not in out
    assert "## **Assistant**" not in out


def test_user_and_assistant_turns_rendered():
    msgs = [
        {"role": "user", "content": "what is X?"},
        {"role": "assistant", "content": "X is Y.", "sources": []},
    ]
    out = conversation_as_markdown(msgs)
    assert "## **You**" in out
    assert "what is X?" in out
    assert "## **Assistant**" in out
    assert "X is Y." in out


def test_sources_rendered_as_blockquotes():
    msgs = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "a",
            "sources": [
                {"source": "doc.pdf", "page": 7, "text": "snippet text"},
            ],
        },
    ]
    out = conversation_as_markdown(msgs)
    assert "`doc.pdf`" in out
    assert "page 7" in out
    assert "snippet text" in out


def test_missing_sources_key_is_safe():
    msgs = [{"role": "user", "content": "hi"}]
    out = conversation_as_markdown(msgs)
    assert "hi" in out
