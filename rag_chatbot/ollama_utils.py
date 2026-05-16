from __future__ import annotations

from typing import Optional

import requests

from rag_chatbot.config import OLLAMA_BASE_URL

_EMBED_HINTS = ("embed", "nomic")


def list_installed_models(
    base_url: Optional[str] = None,
    timeout: float = 3.0,
) -> list[str]:
    url = (base_url or OLLAMA_BASE_URL).rstrip("/") + "/api/tags"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        models = resp.json().get("models", []) or []
    except Exception:
        return []

    names = [m.get("name", "") for m in models if m.get("name")]
    return sorted(n for n in names if not any(h in n.lower() for h in _EMBED_HINTS))
