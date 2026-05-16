import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
CHROMA_DIR  = BASE_DIR / "data" / "chroma_db"
UPLOADS_DIR = BASE_DIR / "uploads"

CHROMA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── Ollama ─────────────────────────────────────────────────────────────────
# Env var lets Docker compose point the app at the ollama service container.
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL         = os.getenv("LLM_MODEL", "llama3")
EMBED_MODEL       = os.getenv("EMBED_MODEL", "nomic-embed-text")

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 50

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K             = 5                 # chunks returned per query
USE_MMR           = True              # Max Marginal Relevance (diversity)
MMR_FETCH_K       = 20                # candidate pool for MMR

# Hybrid search (BM25 keyword + vector semantic)
USE_HYBRID            = False         # toggle in UI; fuses BM25 + vector results
HYBRID_VECTOR_WEIGHT  = 0.6           # weight for vector score (BM25 gets 1 - this)
HYBRID_FETCH_K        = 20            # candidates from each retriever before fusion

# Cross-encoder re-ranking
USE_RERANK        = False             # rerank top results with a cross-encoder
RERANK_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_FETCH_K    = 20                # fetch this many, rerank, keep top_k

# ── Chat memory ────────────────────────────────────────────────────────────
USE_CHAT_MEMORY   = True              # include previous turns in prompt
MAX_HISTORY_TURNS = 4                 # how many prior (user+assistant) pairs to keep

# ── Streaming ──────────────────────────────────────────────────────────────
USE_STREAMING     = True              # stream tokens to the UI as they generate

# ── ChromaDB ───────────────────────────────────────────────────────────────
COLLECTION_NAME   = "rag_documents"

# ── Supported file types ───────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
