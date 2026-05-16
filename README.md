# Local RAG Chatbot

Fully offline chatbot for asking questions about your own documents
(PDF, DOCX, TXT, MD or URLs). Answers are grounded in your data with
inline citations. Nothing leaves your machine.

```
> What does chapter 3 say about climate change?
-> Climate change accelerated post-1980 [Source: report.pdf, Page 47] ...
```

## Quick start (3 commands)

```bash
make build && make up         # build image and start ollama + app
make pull-light               # pull llama3.2:1b + nomic-embed-text (~1.6 GB)
open http://localhost:8501    # open the UI
```

That's it. Upload a PDF in the sidebar, pick `llama3.2:1b` from the model
dropdown, ask a question.

## Features

Core RAG pipeline:

- Hybrid retrieval (BM25 keyword + dense vector with min-max score fusion)
- Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`)
- Citations with hallucination guard (`[Source: file, Page N]`)
- Streaming token output
- Multi-format ingest: PDF, DOCX, TXT, MD, URLs
- RAGAS evaluation (faithfulness, answer relevancy, context recall)

Chat / UX:

- **Document filter** - multi-select chips in the sidebar to scope retrieval
  to specific source files (uses Chroma `$in` metadata filter)
- **Advanced settings** - adjust temperature, chunk size, chunk overlap, and
  custom system prompt directly in the sidebar
- **Conversation export** - download the current chat as Markdown or JSON
- **Query rewriting** - LLM turns follow-ups like "what about chapter 4?"
  into standalone queries before retrieval
- **Auto-detected model dropdown** - reads installed models from
  Ollama's `/api/tags` so the picker never lies about what's pulled
- **Chat memory** - last N user / assistant turns fed into the prompt

Infra / dev:

- **CLI entrypoint** - `rag-chatbot ingest|ask|list|delete|eval`
- **pyproject.toml** - install with `pip install -e .`
- **Makefile** - `make build / up / down / logs / pull / pull-light / test / eval`
- **Unit tests** - chunking determinism, source filter shape, conversation
  export formatting (no LLM needed)
- **Env-overridable models** - `LLM_MODEL` and `EMBED_MODEL` env vars
- Docker Compose stack: Streamlit app + Ollama, with persistent volumes
  for ChromaDB and uploads

## Project structure

```
local_rag_chatbot/
├── rag_chatbot/
│   ├── __init__.py
│   ├── config.py             # paths, model names, default toggles
│   ├── ingest.py             # extraction + chunking + Chroma
│   ├── rag_chain.py          # retrieval + prompt + LLM
│   ├── hybrid_retriever.py   # BM25 + vector score fusion
│   ├── reranker.py           # cross-encoder rerank
│   ├── exporters.py          # markdown conversation export
│   ├── ollama_utils.py       # /api/tags helper
│   ├── evaluate.py           # RAGAS evaluation
│   ├── cli.py                # argparse-based CLI
│   └── app.py                # Streamlit UI
├── tests/
│   ├── test_chunking.py
│   ├── test_source_filter.py
│   └── test_exporters.py
├── data/chroma_db/           # persisted vector store (gitignored)
├── uploads/                  # uploaded files (gitignored)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── Makefile
└── README.md
```

## Run with Docker

```bash
# 1. Build and start ollama + app
make build
make up

# 2. Pull models inside the ollama container
make pull          # llama3 (4.7 GB) + nomic-embed-text (280 MB)
# or, on a low-RAM host (< 8 GB Docker memory):
make pull-light    # llama3.2:1b (1.3 GB) + nomic-embed-text

# 3. Open the UI
open http://localhost:8501

# Tail / stop
make logs
make down
```

Inside the UI:

1. Upload a PDF / DOCX / TXT / MD, or paste a URL
2. (Optional) Tick documents in **"Filter retrieval to selected docs"** to
   scope the query
3. (Optional) Open **"Advanced settings"** to change temperature / chunk
   size / system prompt
4. Ask a question

## Run locally (without Docker)

You need Ollama installed on the host.

```bash
# 1. Pull models via Ollama
ollama pull llama3
ollama pull nomic-embed-text

# 2. Install the package into a venv
python -m venv .venv && source .venv/bin/activate
make install   # pip install -e .

# 3. Run the UI
streamlit run rag_chatbot/app.py
```

## CLI

Once installed, the same operations are available as a CLI:

```bash
# Ingest files or URLs (idempotent - re-ingesting skips existing chunks)
rag-chatbot ingest path/to/file.pdf
rag-chatbot ingest https://example.com/article

# Ask a question
rag-chatbot ask "what does chapter 3 say about X?"
rag-chatbot ask "summarise the main topics" --model llama3.2:1b --top_k 8

# Manage the corpus
rag-chatbot list
rag-chatbot delete file.pdf

# Evaluation
rag-chatbot eval --create_sample    # writes test_questions.json template
rag-chatbot eval --qa_path test_questions.json
```

## Tests

```bash
make test          # runs pytest inside the running app container
make test-local    # if you've already installed deps in a local venv
```

## Evaluation (RAGAS)

```bash
# 1. Generate a QA template, fill it with real Q + ground truth pairs
rag-chatbot eval --create_sample

# 2. Run evaluation against the indexed corpus
make eval
```

Results are written to `ragas_results.csv`.

## Configuration

Most defaults live in `rag_chatbot/config.py`. The two model names are
also overridable via env vars (useful in `docker-compose.yml`):

```bash
LLM_MODEL=llama3.2:1b
EMBED_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
```

## Make targets

| Target | What it does |
| --- | --- |
| `make build` | `docker compose build` |
| `make up` | start ollama + app in the background |
| `make down` | stop containers |
| `make logs` | tail app logs |
| `make ps` | show container state |
| `make shell` | bash inside the app container |
| `make pull` | pull `llama3` + `nomic-embed-text` |
| `make pull-light` | pull `llama3.2:1b` + `nomic-embed-text` |
| `make test` | pytest inside the running app container |
| `make test-local` | pytest in a local venv |
| `make eval` | run RAGAS evaluation in the container |
| `make install` | `pip install -e .` |
| `make clean` | remove ChromaDB cache and pyc files |

## Tech stack

- Ollama (Llama 3 / Llama 3.2)
- nomic-embed-text for embeddings
- LangChain (community + ollama)
- ChromaDB
- rank_bm25
- cross-encoder/ms-marco-MiniLM-L-6-v2
- PyMuPDF, python-docx, BeautifulSoup
- Streamlit
- RAGAS
