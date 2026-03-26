# Battleborn Customer Support AI

This repository contains a first prototype for the customer ticket resolution workflow domain. The final stable path in this repo is RAG-first with `llama3.1:8b` in Ollama:

- Python controls the workflow and tool execution.
- Ollama provides the student model for planning and grounded drafting.
- Mock tools simulate the ticketing system, KB access, posting, status updates, retries, and escalation.
- Every run emits a structured execution log that can later be converted into a fine-tuning dataset.

## What is implemented

- Plan-then-execute agent loop
- Allowed tool registry for the customer support domain
- Retry handling with max two retries per step
- Graceful escalation if a step cannot be resolved
- Clarification-question flow when the ticket is still too vague for a grounded answer
- Evidence-rich escalation summaries with classification, KB attempts, and latest customer context
- JSON execution logs saved to `outputs/`
- KB articles for grounding, retries, and escalation
- SFT export command to convert a run log into a training example
- Interactive chat mode that keeps the conversation going until resolution, escalation, or customer wait-state
- Markdown demo report generation from recent execution logs

## A2A template adaptation

This project is now adapted to an A2A template-style structure.

- Existing runtime remains in `src/ticket_agent/`.
- New A2A wrapper package is in `src/a2a_agent/`.
- A2A entrypoint: `a2a-ticket-agent`
- Docker files added: `Dockerfile`, `docker-compose.yml`

Run the A2A server:

```powershell
pip install -e .
$env:OPENAI_API_KEY = "your-key"
a2a-ticket-agent --host localhost --port 5000
```

## Quick start

1. Create and activate a virtual environment.
2. Install the project:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

3. Create a ticket:

```powershell
ticket-agent create-ticket --subject "Cannot export invoice" --body "The export button fails every time I try it." --priority high
```

4. Run it without Ollama:

```powershell
ticket-agent run --ticket-id TICK-1001 --disable-ollama
```

5. Run it with Ollama and the intended student model:

```powershell
$env:OLLAMA_STUDENT_MODEL = "llama3.1:8b"
ticket-agent run --ticket-id TICK-1001
```

6. Run the interactive customer chat:

```powershell
$env:OLLAMA_STUDENT_MODEL = "llama3.1:8b"
ticket-agent chat --ticket-id TICK-1001
```

7. Generate a judge-facing markdown summary from the latest demo logs:

```powershell
ticket-agent summarize-demo --out outputs\demo_report.md
```

## Create your own ticket

Create a ticket directly from the CLI and store it in the JSON ticket file:

```powershell
ticket-agent create-ticket --subject "Cannot export invoice" --body "The export button fails every time I try it." --priority high
```

If you want a seeded follow-up for scripted demos, add one or more `--scripted-reply` flags:

```powershell
ticket-agent create-ticket --subject "Webhook timeouts" --body "Deliveries are failing." --scripted-reply "We checked the signing secret and it is still failing."
```

If you omit `--subject` or `--body`, the CLI will prompt you for them interactively.

## Expand the knowledge base

The agent now auto-loads every `knowledge_base*.json` file in [data](/C:/Users/parav/Documents/New%20project%202/data). The curated base lives in [knowledge_base.json](/C:/Users/parav/Documents/New%20project%202/data/knowledge_base.json), and imported additions should go into `knowledge_base_extra.json`.

Import additional KB articles from a JSON file:

```powershell
ticket-agent import-kb --source path\to\new_articles.json
```

Each imported file can contain either one article object or a list of article objects using the same schema as the existing KB entries. After import, those articles are loaded automatically in future runs.

Import markdown or text docs and convert them into KB articles:

```powershell
ticket-agent import-kb-docs --source path\to\docs --issue-type integration --information-type product
```

That command scans `.md`, `.markdown`, `.txt`, and `.rst` files, converts each file into one KB article, and writes the generated entries into the expandable KB store.

## Scale up synthetic data

Generate a larger synthetic ticket file:

```powershell
.venv\Scripts\python.exe scripts\generate_synthetic_tickets.py --access-count 30 --billing-count 30 --integration-count 20 --escalation-count 20 --output artifacts\generated_tickets.json
```

Run the teacher exporter against that file:

```powershell
.venv\Scripts\python.exe scripts\generate_teacher_runs.py --all-tickets --tickets-file artifacts\generated_tickets.json --model llama3.1:8b --require-ollama-plan --require-ollama-draft --output-jsonl artifacts\teacher_runs.jsonl
```

## Packaging the stable runtime

If you want a simple Ollama model wrapper for the shipped runtime path, use the root [Modelfile.template](Modelfile.template):

```powershell
Copy-Item Modelfile.template Modelfile -Force
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" create ticket-agent-demo -f .\Modelfile
```

That creates a model based on `llama3.1:8b` with the system prompt tuned for this project.

## Experimental fine-tuning path

The repo still includes data-export, LoRA training, and merge scripts for experimentation, but that is not the stable demo path. The reliable hackathon path is:

1. Use Ollama `llama3.1:8b` as the live runtime model.
2. Use the KB + workflow agent for grounding, retries, and escalation.
3. Use exported teacher traces as distillation artifacts or prompt examples.

If you still want to experiment with offline fine-tuning, the notes are in [training/README.md](training/README.md).
