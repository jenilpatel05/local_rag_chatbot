import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
UPLOADS_DIR = BASE_DIR / "uploads"

CHROMA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

TOP_K = 5
USE_MMR = True
MMR_FETCH_K = 20

USE_HYBRID = False
HYBRID_VECTOR_WEIGHT = 0.6
HYBRID_FETCH_K = 20

USE_RERANK = False
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_FETCH_K = 20

USE_CHAT_MEMORY = True
MAX_HISTORY_TURNS = 4

USE_STREAMING = True

COLLECTION_NAME = "rag_documents"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
