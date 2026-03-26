# Customization Guide

This repository is now aligned to the A2A template file layout.

## What to customize

- Edit `src/agent_toolset.py` for support workflows and RAG policy.
- Adjust system prompt in `src/openai_agent.py`.
- Point `KB_PATH` to your final knowledge base file.
- Set `CLOUD_MODEL` and API key for your provider.

## Model strategy

Cloud-only: `gpt-4.1-mini` (default for speed + quality).

