# Battleborn Customer Support AI (A2A Template)

This project is a clean A2A-template implementation of a customer-support agent using:

- Cloud LLM API (primary)
- Fast Ollama instruct fallback model (`qwen2.5:7b-instruct`)
- Lightweight RAG over local JSON knowledge base

## Template Structure

```
.
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── openai_agent.py
│   ├── openai_agent_executor.py
│   └── agent_toolset.py
├── .gitignore
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Environment Variables

- `OPENAI_API_KEY` (required for cloud model)
- `CLOUD_MODEL` (default: `gpt-4.1-mini`)
- `OLLAMA_HOST` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `qwen2.5:7b-instruct`)
- `KB_PATH` (default: `data/knowledge_base.json`)

## Run

```powershell
pip install -e .
python -m src --host localhost --port 5000
```

