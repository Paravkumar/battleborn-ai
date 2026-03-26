# Battleborn Customer Support AI (A2A Template)

This project is a clean A2A-template implementation for **PS3 Domain 1: Customer Ticket Resolution** using:

- Cloud LLM API (primary)
- Lightweight RAG over local JSON knowledge base

The workflow engine now enforces:
- ordered tool plan (ticket_reader -> knowledge_base_query -> response_composer -> ticket_updater)
- retries with modified arguments (max two retries)
- escalation via `escalation_trigger` when a step cannot be completed
- structured JSON execution log per run in `outputs/`

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

- `NVIDIA_KEY` (required for cloud model)
- `CLOUD_MODEL` (default: `neotron-3-nano`)
- `KB_PATH` (default: `data/knowledge_base.json`)

## Run

```powershell
pip install -e .
python -m src --host localhost --port 5000
```

