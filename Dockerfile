FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY rag_chatbot/ ./rag_chatbot/
COPY tests/ ./tests/
RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/data/chroma_db /app/uploads

EXPOSE 8501

ENV OLLAMA_BASE_URL=http://ollama:11434
ENV PYTHONPATH=/app

CMD ["streamlit", "run", "rag_chatbot/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
