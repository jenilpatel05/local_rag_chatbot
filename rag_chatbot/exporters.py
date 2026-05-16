from __future__ import annotations

from datetime import datetime


def conversation_as_markdown(messages: list[dict]) -> str:
    lines = [
        f"# Conversation export - {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for msg in messages:
        role = "**You**" if msg["role"] == "user" else "**Assistant**"
        lines.append(f"## {role}")
        lines.append("")
        lines.append(msg["content"])
        lines.append("")
        for src in msg.get("sources") or []:
            lines.append(
                f"> `{src['source']}` page {src['page']}: \"{src['text']}...\""
            )
        lines.append("")
    return "\n".join(lines)
