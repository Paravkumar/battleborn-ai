#!/usr/bin/env python3
"""
Data Pipeline Executor - Run the full ticket generation and processing pipeline
Usage: python run_pipeline_direct.py
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Change to project directory
    project_dir = r'C:\Users\parav\Documents\New project 2'
    os.chdir(project_dir)
    
    print("\n" + "="*70)
    print("DATA PIPELINE EXECUTOR")
    print("="*70)
    
    # Step 1: Generate synthetic tickets
    print("\n[1/4] Running synthetic ticket generation...")
    print("-" * 70)
    step1_cmd = [
        '.venv\\Scripts\\python.exe',
        'scripts\\generate_synthetic_tickets.py',
        '--access-count', '1',
        '--billing-count', '1',
        '--integration-count', '1',
        '--escalation-count', '1',
        '--output', 'artifacts\\generated_tickets_more.json'
    ]
    
    result = subprocess.run(step1_cmd, shell=False)
    if result.returncode != 0:
        print(f"\n✗ Step 1 FAILED (exit code {result.returncode})")
        return 1
    print("✓ Step 1 complete: Synthetic tickets generated")
    
    # Step 2: Generate teacher runs
    print("\n[2/4] Running teacher run generation (this will take a while)...")
    print("-" * 70)
    step2_cmd = [
        '.venv\\Scripts\\python.exe',
        'scripts\\generate_teacher_runs.py',
        '--all-tickets',
        '--tickets-file', 'artifacts\\generated_tickets_more.json',
        '--model', 'llama3.1:70b',
        '--require-ollama-plan',
        '--require-ollama-draft',
        '--output-jsonl', 'artifacts\\teacher_runs_more.jsonl'
    ]
    
    result = subprocess.run(step2_cmd, shell=False)
    if result.returncode != 0:
        print(f"\n✗ Step 2 FAILED (exit code {result.returncode})")
        return 1
    print("✓ Step 2 complete: Teacher runs generated")
    
    # Step 3: Append non-empty lines
    print("\n[3/4] Appending non-empty lines to teacher_runs.jsonl...")
    print("-" * 70)
    
    source_file = Path('artifacts') / 'teacher_runs_more.jsonl'
    dest_file = Path('artifacts') / 'teacher_runs.jsonl'
    
    if not source_file.exists():
        print(f"✗ Source file not found: {source_file}")
        return 1
    
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Read non-empty lines and append
    non_empty_count = 0
    with open(source_file, 'r', encoding='utf-8') as src:
        with open(dest_file, 'a', encoding='utf-8') as dst:
            for line in src:
                if line.strip():
                    dst.write(line if line.endswith('\n') else line + '\n')
                    non_empty_count += 1
    
    print(f"✓ Step 3 complete: Appended {non_empty_count} non-empty lines")
    
    # Step 4: Count and report
    print("\n[4/4] Final line counts...")
    print("-" * 70)
    
    # Count non-empty lines in each file
    def count_non_empty_lines(filepath):
        if not filepath.exists():
            return 0
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    
    more_count = count_non_empty_lines(source_file)
    total_count = count_non_empty_lines(dest_file)
    
    print(f"Lines in artifacts\\teacher_runs_more.jsonl: {more_count}")
    print(f"Total lines in artifacts\\teacher_runs.jsonl: {total_count}")
    print("✓ Step 4 complete: Counts verified")
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
