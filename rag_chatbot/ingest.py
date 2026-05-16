from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse

import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from rag_chatbot.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    SUPPORTED_EXTENSIONS,
)


def _extract_pdf(pdf_path: Path) -> list[dict]:
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text").strip()
        if not text:
            continue
        pages.append({
            "text": text,
            "page": page_num + 1,
            "source": pdf_path.name,
        })
    doc.close()
    return pages


def _extract_docx(docx_path: Path) -> list[dict]:
    try:
        from docx import Document
    except ImportError as e:
        raise ImportError(
            "python-docx is required for DOCX support. "
            "Install with: pip install python-docx"
        ) from e

    document = Document(str(docx_path))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs).strip()
    if not text:
        return []
    return [{"text": text, "page": 1, "source": docx_path.name}]


def _extract_text_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    return [{"text": text, "page": 1, "source": path.name}]


def _extract_url(url: str) -> list[dict]:
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise ImportError(
            "requests and beautifulsoup4 are required for URL support."
        ) from e

    resp = requests.get(url, timeout=20, headers={"User-Agent": "local-rag-chatbot"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    text = soup.get_text(separator="\n").strip()
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not text:
        return []

    parsed = urlparse(url)
    source = (parsed.netloc + parsed.path).rstrip("/") or url
    return [{"text": text, "page": 1, "source": source}]


def extract_pages(path_or_url: str | Path) -> list[dict]:
    if isinstance(path_or_url, str) and path_or_url.lower().startswith(("http://", "https://")):
        return _extract_url(path_or_url)

    path = Path(path_or_url)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix in {".txt", ".md"}:
        return _extract_text_file(path)
    raise ValueError(
        f"Unsupported file type: {suffix}. Supported: {sorted(SUPPORTED_EXTENSIONS)} or URLs."
    )


def chunk_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for i, split in enumerate(splits):
            uid = hashlib.md5(
                f"{page['source']}|{page['page']}|{i}".encode()
            ).hexdigest()
            chunks.append({
                "text": split,
                "metadata": {
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": i,
                    "chunk_id": uid,
                },
            })
    return chunks


def get_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def ingest_source(
    path_or_url: str | Path,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> tuple[int, int]:
    pages = extract_pages(path_or_url)
    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not chunks:
        return 0, 0

    vs = get_vectorstore()

    existing_ids: set[str] = set()
    try:
        existing = vs.get(include=[])
        existing_ids = set(existing.get("ids", []))
    except Exception:
        pass

    new_chunks = [c for c in chunks if c["metadata"]["chunk_id"] not in existing_ids]

    if new_chunks:
        vs.add_texts(
            texts=[c["text"] for c in new_chunks],
            metadatas=[c["metadata"] for c in new_chunks],
            ids=[c["metadata"]["chunk_id"] for c in new_chunks],
        )

    return len(pages), len(new_chunks)


def ingest_pdf(path: str | Path) -> tuple[int, int]:
    return ingest_source(path)


def list_ingested_sources() -> list[str]:
    vs = get_vectorstore()
    try:
        result = vs.get(include=["metadatas"])
        sources = {m["source"] for m in result["metadatas"] if "source" in m}
        return sorted(sources)
    except Exception:
        return []


def delete_source(filename: str) -> int:
    vs = get_vectorstore()
    try:
        result = vs.get(include=["metadatas"])
        ids_to_delete = [
            rid for rid, meta in zip(result["ids"], result["metadatas"])
            if meta.get("source") == filename
        ]
        if ids_to_delete:
            vs._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
    except Exception:
        return 0


def get_all_chunks() -> list[dict]:
    vs = get_vectorstore()
    try:
        result = vs.get(include=["documents", "metadatas"])
        return [
            {"text": t, "metadata": m}
            for t, m in zip(result["documents"], result["metadatas"])
        ]
    except Exception:
        return []


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file_or_url> [file_or_url ...]")
        sys.exit(1)
    for arg in sys.argv[1:]:
        print(f"[ingest] {arg} ...", end=" ", flush=True)
        try:
            pages, chunks = ingest_source(arg)
            print(f"{pages} pages -> {chunks} new chunks stored")
        except Exception as e:
            print(f"FAILED: {e}")
