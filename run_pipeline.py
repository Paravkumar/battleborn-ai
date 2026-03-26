#!/usr/bin/env python3
"""
Orchestrate the data pipeline to generate teacher training data.
"""
import subprocess
import sys
import os
from pathlib import Path

# Change to repo directory
os.chdir("C:\\Users\\parav\\Documents\\New project 2")

def run_command(cmd_list, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd_list)}")
    print()
    
    try:
        result = subprocess.run(cmd_list, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"ERROR: {description} failed with return code {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"ERROR: {description} raised exception: {e}")
        return False

# Step 1: Generate synthetic tickets
if not run_command([
    ".venv\\Scripts\\python.exe",
    "scripts\\generate_synthetic_tickets.py",
    "--access-count", "1",
    "--billing-count", "1",
    "--integration-count", "1",
    "--escalation-count", "1",
    "--output", "artifacts\\generated_tickets_more.json"
], "Step 1: Generate synthetic tickets"):
    sys.exit(1)

# Step 2: Generate teacher runs
if not run_command([
    ".venv\\Scripts\\python.exe",
    "scripts\\generate_teacher_runs.py",
    "--all-tickets",
    "--tickets-file", "artifacts\\generated_tickets_more.json",
    "--model", "llama3.1:70b",
    "--require-ollama-plan",
    "--require-ollama-draft",
    "--output-jsonl", "artifacts\\teacher_runs_more.jsonl"
], "Step 2: Generate teacher runs (using llama3.1:70b)"):
    sys.exit(1)

# Step 3: Append non-empty lines
print(f"\n{'='*60}")
print("Step 3: Append non-empty lines to teacher_runs.jsonl")
print(f"{'='*60}")

try:
    # Read generated file
    generated_file = Path("artifacts/teacher_runs_more.jsonl")
    if not generated_file.exists():
        print(f"ERROR: Generated file {generated_file} not found!")
        sys.exit(1)
    
    generated_lines = [line for line in generated_file.read_text().splitlines() if line.strip()]
    print(f"Read {len(generated_lines)} non-empty lines from artifacts/teacher_runs_more.jsonl")
    
    # Append to main file
    output_file = Path("artifacts/teacher_runs.jsonl")
    with open(output_file, "a") as f:
        for line in generated_lines:
            f.write(line + "\n")
    
    print(f"Appended to artifacts/teacher_runs.jsonl")
    
except Exception as e:
    print(f"ERROR during append: {e}")
    sys.exit(1)

# Step 4: Print counts
print(f"\n{'='*60}")
print("Step 4: Final counts")
print(f"{'='*60}")

try:
    more_count = len([line for line in Path("artifacts/teacher_runs_more.jsonl").read_text().splitlines() if line.strip()])
    total_count = len([line for line in Path("artifacts/teacher_runs.jsonl").read_text().splitlines() if line.strip()])
    
    print(f"Lines in artifacts/teacher_runs_more.jsonl: {more_count}")
    print(f"Total lines in artifacts/teacher_runs.jsonl: {total_count}")
    print(f"\nPipeline completed successfully!")
    
except Exception as e:
    print(f"ERROR reading counts: {e}")
    sys.exit(1)
