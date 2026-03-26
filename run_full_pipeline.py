#!/usr/bin/env python3
"""
Execute the full data pipeline: generate synthetic tickets, generate teacher runs with llama3.1:70b,
append to existing training data, and report counts.
"""

import subprocess
import sys
import os
from pathlib import Path

# Ensure we're in the right directory
repo_root = Path(r"C:\Users\parav\Documents\New project 2")
os.chdir(repo_root)

python_exe = repo_root / ".venv" / "Scripts" / "python.exe"

def run_step(step_num, description, cmd):
    """Run a command step and handle errors."""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\nERROR: Step {step_num} failed with exit code {result.returncode}")
        return False
    return True

def count_lines(filepath):
    """Count non-empty lines in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return len([line for line in f if line.strip()])
    except FileNotFoundError:
        return 0

# Step 1: Generate synthetic tickets
if not run_step(1, "Generate synthetic tickets", [
    str(python_exe),
    "scripts/generate_synthetic_tickets.py",
    "--access-count", "1",
    "--billing-count", "1", 
    "--integration-count", "1",
    "--escalation-count", "1",
    "--output", "artifacts/generated_tickets_more.json"
]):
    sys.exit(1)

# Step 2: Generate teacher runs with llama3.1:70b
if not run_step(2, "Generate teacher runs (llama3.1:70b) - SLOW, please wait...", [
    str(python_exe),
    "scripts/generate_teacher_runs.py",
    "--all-tickets",
    "--tickets-file", "artifacts/generated_tickets_more.json",
    "--model", "llama3.1:70b",
    "--require-ollama-plan",
    "--require-ollama-draft",
    "--output-jsonl", "artifacts/teacher_runs_more.jsonl"
]):
    sys.exit(1)

# Step 3: Append non-empty lines
print(f"\n{'='*70}")
print("STEP 3: Append non-empty lines to teacher_runs.jsonl")
print(f"{'='*70}")

more_file = repo_root / "artifacts" / "teacher_runs_more.jsonl"
total_file = repo_root / "artifacts" / "teacher_runs.jsonl"

if more_file.exists():
    with open(more_file, 'r', encoding='utf-8') as src:
        lines_to_append = [line for line in src if line.strip()]
    
    print(f"Read {len(lines_to_append)} non-empty lines from {more_file.name}")
    
    with open(total_file, 'a', encoding='utf-8') as dst:
        for line in lines_to_append:
            if not line.endswith('\n'):
                line = line + '\n'
            dst.write(line)
    
    print(f"Appended to {total_file.name}")
else:
    print(f"Warning: {more_file.name} not found")
    lines_to_append = []

# Step 4: Print final counts
print(f"\n{'='*70}")
print("STEP 4: Final Counts")
print(f"{'='*70}")

more_count = count_lines(more_file)
total_count = count_lines(total_file)

print(f"Lines in artifacts/teacher_runs_more.jsonl: {more_count}")
print(f"Total lines in artifacts/teacher_runs.jsonl: {total_count}")
print(f"\n✓ Pipeline completed successfully!")

sys.exit(0)
