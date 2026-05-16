# Local RAG Chatbot (Llama 3)

This project is a **fully offline chatbot** that lets you ask questions from your own PDFs.

Instead of manually going through documents, you can just upload them and ask questions like:

> “What does chapter 3 say about climate change?”

It will answer **based only on your documents** and also show **where the answer came from (page + content)**.

Everything runs locally using Llama 3 — no APIs, no cloud, no data sharing.

---

## What it does

* Upload multiple PDF files
* Reads and understands the content
* Lets you ask questions in simple English
* Gives accurate answers with **citations**
* Runs completely on your system

---

## How it works (simple idea)

This project uses **RAG (Retrieval-Augmented Generation)**.

Instead of the model guessing answers:

1. It first **searches your documents**
2. Picks the most relevant parts
3. Sends that as context to the model
4. Then generates the answer

So the answers are **grounded in your data**, not random.

---

## Flow

**When you upload PDFs:**

```
PDF → Text → Chunks → Embeddings → Stored in DB
```

**When you ask a question:**

```
Question → Search relevant chunks → Send to LLM → Answer + Sources
```

---

## Tech Stack

* **Ollama** – to run Llama 3 locally
* **Llama 3 (8B)** – main LLM
* **LangChain** – handles RAG pipeline
* **ChromaDB** – vector database
* **Sentence Transformers** – embeddings
* **PyMuPDF / PyPDF** – PDF parsing
* **Streamlit** – UI

---

## Requirements

* 16GB RAM recommended (works on 8GB but slower)
* ~5GB storage for model
* GPU optional (CPU works fine)

---

## Setup

### 1. Install Ollama and pull model

```bash
ollama pull llama3
```

---

### 2. Create virtual environment

```bash
python -m venv venv
```

Activate:

```bash
# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install langchain langchain-community chromadb
pip install sentence-transformers pypdf pymupdf
pip install streamlit
```

---

## ▶️ Run the project

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
local-rag-chatbot/
├── app.py
├── ingest.py
├── chains.py
├── chroma_db/
├── uploads/
├── requirements.txt
└── README.md
```

---

## Note

Everything runs locally.
No data is sent anywhere.

---
