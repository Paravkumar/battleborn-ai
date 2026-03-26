#!/usr/bin/env python3
"""
Execute the data pipeline: Generate synthetic tickets, teacher runs, and append results.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"=== {description} ===")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, cwd=os.getcwd())
    
    if result.returncode != 0:
        print(f"\n❌ {description} FAILED with exit code {result.returncode}")
        return False
    
    print(f"\n✓ {description} completed successfully")
    return True

def main():
    """Execute the full pipeline."""
    os.chdir('C:\\Users\\parav\\Documents\\New project 2')
    
    # Step 1: Generate synthetic tickets
    cmd1 = '.venv\\Scripts\\python.exe scripts\\generate_synthetic_tickets.py --access-count 1 --billing-count 1 --integration-count 1 --escalation-count 1 --output artifacts\\generated_tickets_more.json'
    if not run_command(cmd1, "Step 1: Generating synthetic tickets"):
        sys.exit(1)
    
    # Step 2: Generate teacher runs (this will be slow)
    cmd2 = '.venv\\Scripts\\python.exe scripts\\generate_teacher_runs.py --all-tickets --tickets-file artifacts\\generated_tickets_more.json --model llama3.1:70b --require-ollama-plan --require-ollama-draft --output-jsonl artifacts\\teacher_runs_more.jsonl'
    if not run_command(cmd2, "Step 2: Generating teacher runs with llama3.1:70b (this may take a while)"):
        sys.exit(1)
    
    # Step 3: Append non-empty lines to artifacts\teacher_runs.jsonl
    print(f"\n{'='*60}")
    print("=== Step 3: Appending non-empty lines ===")
    print(f"{'='*60}")
    
    source_file = Path('artifacts') / 'teacher_runs_more.jsonl'
    dest_file = Path('artifacts') / 'teacher_runs.jsonl'
    
    if not source_file.exists():
        print(f"❌ Source file {source_file} not found")
        sys.exit(1)
    
    # Ensure destination directory exists
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Read non-empty lines from source and append to destination
    with open(source_file, 'r', encoding='utf-8') as f:
        non_empty_lines = [line for line in f if line.strip()]
    
    with open(dest_file, 'a', encoding='utf-8') as f:
        for line in non_empty_lines:
            f.write(line)
            if not line.endswith('\n'):
                f.write('\n')
    
    print(f"✓ Appended {len(non_empty_lines)} non-empty lines to {dest_file}")
    
    # Step 4: Count and print
    print(f"\n{'='*60}")
    print("=== Step 4: Final Counts ===")
    print(f"{'='*60}")
    
    with open(source_file, 'r', encoding='utf-8') as f:
        more_count = len([line for line in f if line.strip()])
    
    with open(dest_file, 'r', encoding='utf-8') as f:
        total_count = len([line for line in f if line.strip()])
    
    print(f"\nLines in artifacts\\teacher_runs_more.jsonl: {more_count}")
    print(f"Total lines in artifacts\\teacher_runs.jsonl: {total_count}")
    
    print(f"\n{'='*60}")
    print("=== ✓ Pipeline completed successfully ===")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
