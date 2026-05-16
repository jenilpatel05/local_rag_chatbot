.PHONY: build up down logs ps shell pull pull-light test test-local eval install clean

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f app

ps:
	docker compose ps

shell:
	docker compose exec app /bin/bash

pull:
	docker compose exec ollama ollama pull llama3
	docker compose exec ollama ollama pull nomic-embed-text

pull-light:
	docker compose exec ollama ollama pull llama3.2:1b
	docker compose exec ollama ollama pull nomic-embed-text

test:
	docker compose exec app pytest

test-local:
	pytest

eval:
	docker compose exec app python -m rag_chatbot.evaluate --qa_path test_questions.json

install:
	pip install -e .

clean:
	rm -rf data/chroma_db .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
