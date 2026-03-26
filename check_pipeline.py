#!/usr/bin/env python3
"""Check pipeline status and run if needed."""
import subprocess
import sys
from pathlib import Path

def count_lines(filepath):
    """Count non-empty lines in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line for line in f if line.strip()]
        return len(lines)
    except FileNotFoundError:
        return 0

# Go to repo
repo_dir = Path("C:/Users/parav/Documents/New project 2")
repo_dir_win = Path(r"C:\Users\parav\Documents\New project 2")
print(f"Working in: {repo_dir}")

# Check if files exist
more_file = repo_dir_win / "artifacts" / "teacher_runs_more.jsonl"
total_file = repo_dir_win / "artifacts" / "teacher_runs.jsonl"

print(f"teacher_runs_more.jsonl exists: {more_file.exists()}")
print(f"teacher_runs.jsonl exists: {total_file.exists()}")

# If files are empty or don't exist, run the pipeline
more_count = count_lines(more_file) if more_file.exists() else 0
total_count = count_lines(total_file) if total_file.exists() else 0

print(f"\nCurrent counts:")
print(f"  teacher_runs_more.jsonl: {more_count} lines")
print(f"  teacher_runs.jsonl: {total_count} lines")

if more_count == 0:
    print("\nRunning pipeline (more file is empty)...")
    venv_python = repo_dir_win / ".venv" / "Scripts" / "python.exe"
    
    print("\n" + "="*70)
    print("STEP 1: Generate synthetic tickets")
    print("="*70)
    result = subprocess.run([
        str(venv_python),
        str(repo_dir_win / "scripts" / "generate_synthetic_tickets.py"),
        "--access-count", "1",
        "--billing-count", "1",
        "--integration-count", "1",
        "--escalation-count", "1",
        "--output", "artifacts/generated_tickets_more.json"
    ], cwd=str(repo_dir_win))
    
    if result.returncode != 0:
        print(f"ERROR in step 1: {result.returncode}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("STEP 2: Generate teacher runs with llama3.1:70b (this may take a while)")
    print("="*70)
    result = subprocess.run([
        str(venv_python),
        str(repo_dir_win / "scripts" / "generate_teacher_runs.py"),
        "--all-tickets",
        "--tickets-file", "artifacts/generated_tickets_more.json",
        "--model", "llama3.1:70b",
        "--require-ollama-plan",
        "--require-ollama-draft",
        "--output-jsonl", "artifacts/teacher_runs_more.jsonl"
    ], cwd=str(repo_dir_win))
    
    if result.returncode != 0:
        print(f"ERROR in step 2: {result.returncode}")
        sys.exit(1)
    
    # Recount after generation
    more_count = count_lines(more_file)
    print(f"\nGenerated {more_count} examples in teacher_runs_more.jsonl")
    
    print("\n" + "="*70)
    print("STEP 3: Append non-empty lines to teacher_runs.jsonl")
    print("="*70)
    
    # Append non-empty lines
    if more_count > 0:
        with open(more_file, 'r', encoding='utf-8') as src:
            lines = [line for line in src if line.strip()]
        
        with open(total_file, 'a', encoding='utf-8') as dst:
            for line in lines:
                dst.write(line if line.endswith('\n') else line + '\n')
        
        print(f"Appended {len(lines)} lines to teacher_runs.jsonl")
    else:
        print("No lines to append (more file is empty)")

# Final counts
print("\n" + "="*70)
print("FINAL COUNTS")
print("="*70)
more_final = count_lines(more_file)
total_final = count_lines(total_file)

print(f"Lines in artifacts/teacher_runs_more.jsonl: {more_final}")
print(f"Total lines in artifacts/teacher_runs.jsonl: {total_final}")
