# Data Pipeline Execution Instructions

## Current Status
- ✓ Synthetic tickets generated: `artifacts/generated_tickets_more.json` (4 tickets)
- ❌ Teacher runs not yet generated: `artifacts/teacher_runs_more.jsonl` (empty)
- ❌ Main teacher runs file: `artifacts/teacher_runs.jsonl` (empty)

## To Run the Pipeline

Execute this command in your terminal/command prompt:

```cmd
cd C:\Users\parav\Documents\New project 2
.venv\Scripts\python.exe run_full_pipeline.py
```

Or use the batch wrapper:

```cmd
C:\Users\parav\Documents\New project 2\execute_pipeline.cmd
```

## What the Pipeline Does

### Step 1: Generate Synthetic Tickets
Command:
```
.venv\Scripts\python.exe scripts\generate_synthetic_tickets.py --access-count 1 --billing-count 1 --integration-count 1 --escalation-count 1 --output artifacts\generated_tickets_more.json
```
Status: ✓ Complete

### Step 2: Generate Teacher Runs with llama3.1:70b
Command:
```
.venv\Scripts\python.exe scripts\generate_teacher_runs.py --all-tickets --tickets-file artifacts\generated_tickets_more.json --model llama3.1:70b --require-ollama-plan --require-ollama-draft --output-jsonl artifacts\teacher_runs_more.jsonl
```
Status: ❌ Pending - **This step will take several minutes as it runs Ollama**

### Step 3: Append Non-Empty Lines
- Reads all non-empty lines from `artifacts/teacher_runs_more.jsonl`
- Appends them to `artifacts/teacher_runs.jsonl`
- Creates the file if it doesn't exist

### Step 4: Report Final Counts
Prints:
- Number of lines in `artifacts/teacher_runs_more.jsonl`
- Total number of lines in `artifacts/teacher_runs.jsonl`

## Prepared Scripts

1. **run_full_pipeline.py** - Main pipeline orchestrator (recommended)
2. **execute_pipeline.cmd** - Windows batch wrapper
3. **run_pipeline.bat** - Alternative batch script
4. **run_pipeline.py** - Alternative pipeline executor
5. **check_status.py** - Quick status checker

## Expected Output

After running the pipeline, you should see:
```
Lines in artifacts/teacher_runs_more.jsonl: [count]
Total lines in artifacts/teacher_runs.jsonl: [total_count]
```

The counts depend on how many teacher traces completed successfully with the llama3.1:70b model.
