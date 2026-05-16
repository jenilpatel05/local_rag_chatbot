# Dockerfile — Local RAG Chatbot
#
# This image runs the Streamlit app only. Ollama runs as a SEPARATE service
# (see docker-compose.yml) because Ollama needs GPU access and a different
# lifecycle. We connect to it via OLLAMA_BASE_URL.
#
# Build:   docker build -t local-rag-chatbot .
# Run:     docker compose up

FROM python:3.11-slim

# System deps for PyMuPDF + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Persisted data dirs (mounted as volumes in compose)
RUN mkdir -p /app/data/chroma_db /app/uploads

EXPOSE 8501

# Ollama base URL is injected from compose so the app can reach the
# ollama container by service name.
ENV OLLAMA_BASE_URL=http://ollama:11434

# Streamlit needs --server.address=0.0.0.0 to be reachable from outside the container
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
