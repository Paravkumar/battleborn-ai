# A2A Integration Notes

This repository has been adapted to the A2A agent template structure by adding an `a2a_agent` package under `src/`.

## What was wired

- `src/a2a_agent/__main__.py`: A2A server entrypoint.
- `src/a2a_agent/openai_agent.py`: agent wiring and system prompt.
- `src/a2a_agent/openai_agent_executor.py`: OpenAI function-calling executor.
- `src/a2a_agent/agent_toolset.py`: bridge tool (`resolve_ticket`) into existing `ticket_agent` workflow.

## Runtime requirements

- `OPENAI_API_KEY` for A2A/OpenAI orchestration.
- Ollama running locally if you want the existing `ticket_agent` model flow (`OLLAMA_HOST`, `OLLAMA_STUDENT_MODEL`).

## Launch

```powershell
pip install -e .
a2a-ticket-agent --host localhost --port 5000
```

