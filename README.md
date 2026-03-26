# Battleborn Customer Support AI (NVIDIA NIM)

This project is an AI-powered automated workflow implementation for **Problem Statement 3: Domain 1 (Customer Ticket Resolution)**, running purely on NVIDIA's NIM API architecture.

## Features
- Full compliance with PS3 Domain 1 rubric (Adaptive Workflow Orchestration Agent).
- Ordered tool execution (`ticket_reader` -> `knowledge_base_query` -> `response_composer` -> `ticket_updater`).
- Intelligent replanning on failures with parameter mutation (up to 2 retries per step).
- Graceful escalation using the `escalation_trigger` tool upon consecutive failures.
- Structured JSON execution log (`ExecutionLog`) produced in `outputs/` summarizing step outcomes, retries, and errors.
- Connects directly to `integrate.api.nvidia.com` for fast OpenAI-compatible Chat Completions using Nemotron models.

## Template Structure

```text
.
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── nvidia_agent.py
│   ├── nvidia_agent_executor.py
│   └── agent_toolset.py
├── .gitignore
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Environment Setup

Inside `docker-compose.yml`, supply:
- `NVIDIA_API_KEY` (Required. Pull your API key from build.nvidia.com).
- `CLOUD_MODEL` (Default: `nvidia/nemotron-3-nano-30b-a3b` or `nvidia/nemotron-4-340b-instruct`)
- `KB_PATH` (Default: `data/knowledge_base.json`)

## How to Run

1. Start the Docker container
```powershell
docker compose up --build
```

2. Once it says Uvicorn is running, send a test via PowerShell:
```powershell
Invoke-RestMethod -Uri "http://localhost:5000/agent/message" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"message": "Customer says their earbuds are not pairing over bluetooth."}'
```
