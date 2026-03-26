# Fine-Tuning Notes

This training path is experimental. The stable hackathon demo path in this repo is the RAG-first runtime with `llama3.1:8b` in Ollama.

The current codebase already emits execution logs with:

- the ticket snapshot
- the chosen plan
- every tool call attempt
- retries
- the final response or escalation reason

That is enough to bootstrap an SFT dataset for offline experiments, prompt distillation, or future student-model work.

## Recommended loop

1. Generate many synthetic tickets across your supported issue types.
2. Run the strongest available teacher model to create high-quality plans and replies.
3. Validate the traces with Python rules.
4. Export the approved logs with `ticket-agent export-sft`.
5. Optionally fine-tune with LoRA using TRL or Unsloth.
6. Optionally merge the adapter into the base model for offline testing.

For the final demo, stop after step 4 and keep serving `llama3.1:8b` through Ollama.

## Commands

Generate a larger synthetic ticket set first:

```powershell
.venv\Scripts\python.exe scripts\generate_synthetic_tickets.py --access-count 30 --billing-count 30 --integration-count 20 --escalation-count 20 --output artifacts\generated_tickets.json
```

Generate teacher traces and a JSONL SFT dataset:

```powershell
.venv\Scripts\python.exe scripts\generate_teacher_runs.py --all-tickets --tickets-file artifacts\generated_tickets.json --model llama3.1:70b --require-ollama-plan --require-ollama-draft --output-jsonl artifacts\teacher_runs.jsonl
```

Train a LoRA adapter from the exported dataset:

```powershell
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install ".[train]"
.venv\Scripts\python.exe scripts\train_lora.py --dataset artifacts\teacher_runs.jsonl --output-dir artifacts\lora-ticket-agent
```

Merge the adapter into a full model folder for offline experiments:

```powershell
.venv\Scripts\python.exe scripts\merge_lora.py --adapter-dir artifacts\lora-ticket-agent --output-dir artifacts\merged-ticket-student
```

Stable Ollama packaging for the shipped runtime uses the root `Modelfile.template`:

```text
FROM llama3.1:8b
SYSTEM You are a customer ticket resolution agent. Use only the provided tools and produce grounded support replies.
PARAMETER temperature 0.1
```

If traces are being written with `plan_source=fallback` or `response_source=fallback`, do not use them as teacher data. That means the local Ollama service or the requested model was unavailable, so the exporter dropped back to deterministic behavior.

Escalated runs are still valid teacher traces even when `response_source=none`, because the workflow may escalate before any customer reply is drafted.

If you still want to fine-tune a gated Llama checkpoint, first request access in your browser and then authenticate locally:

```powershell
.venv\Scripts\hf.exe auth login
```

Then rerun training with:

```powershell
.venv\Scripts\python.exe scripts\train_lora.py --model-id meta-llama/Llama-3.1-8B-Instruct --dataset artifacts\teacher_runs.jsonl --output-dir artifacts\lora-ticket-agent
```

## Dataset target

Each example should preserve tool-calling behavior, not only the final reply:

- user ticket and goal
- available tool schemas
- assistant plan
- assistant tool calls
- tool outputs
- final grounded answer or escalation

## Suggested next files

- `scripts/generate_teacher_runs.py` to batch-generate distillation traces
- `scripts/train_lora.py` to train an experimental student adapter
- `scripts/merge_lora.py` to merge the adapter into full weights for offline testing
- `Modelfile.template` to package the stable `llama3.1:8b` runtime into Ollama
