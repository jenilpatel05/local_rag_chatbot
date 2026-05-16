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

Upload a PDF in the sidebar, pick `llama3.2:1b` from the model dropdown,
ask a question.

## Features

Core RAG pipeline:

- Hybrid retrieval (BM25 keyword + dense vector with min-max score fusion)
- Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`)
- Citations with hallucination guard (`[Source: file, Page N]`)
- Streaming token output
- Multi-format ingest: PDF, DOCX, TXT, MD, URLs
- RAGAS evaluation (faithfulness, answer relevancy, context recall)

Chat / UX:

- Document filter: multi-select chips in the sidebar to scope retrieval
  to specific source files (Chroma `$in` metadata filter)
- Advanced settings: temperature, chunk size, chunk overlap, custom
  system prompt
- Conversation export to Markdown or JSON
- Query rewriting for follow-up questions
- Auto-detected model dropdown (reads from Ollama's `/api/tags`)
- Chat memory across the last N turns

Infra / dev:

- `rag-chatbot` CLI: ingest, ask, list, delete, eval
- `pyproject.toml` so the package installs with `pip install -e .`
- Makefile for build / up / down / pull / test / eval
- Unit tests with pytest (no LLM required)
- `LLM_MODEL` and `EMBED_MODEL` overridable from the environment
- Docker Compose with Streamlit app + Ollama, persistent volumes for
  ChromaDB and uploads

## Project structure

```
local_rag_chatbot/
├── rag_chatbot/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py
│   ├── rag_chain.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   ├── exporters.py
│   ├── ollama_utils.py
│   ├── evaluate.py
│   ├── cli.py
│   └── app.py
├── tests/
│   ├── test_chunking.py
│   ├── test_source_filter.py
│   └── test_exporters.py
├── data/chroma_db/
├── uploads/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── Makefile
└── README.md
```

## Run with Docker

```bash
make build
make up

make pull          # llama3 (4.7 GB) + nomic-embed-text (280 MB)
# or, on a low-RAM host (< 8 GB Docker memory):
make pull-light    # llama3.2:1b (1.3 GB) + nomic-embed-text

open http://localhost:8501

make logs
make down
```

Inside the UI:

1. Upload a PDF / DOCX / TXT / MD, or paste a URL
2. (Optional) Tick documents in "Filter retrieval to selected docs" to
   scope the query
3. (Optional) Open "Advanced settings" to change temperature, chunk
   size, or the system prompt
4. Ask a question

## Run locally (without Docker)

```bash
ollama pull llama3
ollama pull nomic-embed-text

python -m venv .venv && source .venv/bin/activate
make install

streamlit run rag_chatbot/app.py
```

## CLI

```bash
rag-chatbot ingest path/to/file.pdf
rag-chatbot ingest https://example.com/article
rag-chatbot ask "what does chapter 3 say about X?"
rag-chatbot ask "summarise the main topics" --model llama3.2:1b --top_k 8
rag-chatbot list
rag-chatbot delete file.pdf
rag-chatbot eval --create_sample
rag-chatbot eval --qa_path test_questions.json
```

## Tests

```bash
make test          # pytest inside the running app container
make test-local    # pytest in a local venv
```

## Evaluation (RAGAS)

```bash
rag-chatbot eval --create_sample    # writes test_questions.json template
make eval                           # runs RAGAS against the corpus
```

Results land in `ragas_results.csv`.

## Configuration

Most defaults live in `rag_chatbot/config.py`. The two model names are
also overridable via env vars:

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
