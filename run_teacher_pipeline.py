#!/usr/bin/env python3
"""
Teacher Run Pipeline - Complete 3-step pipeline
Steps:
  1. Generate teacher runs with llama3.1:70b
  2. Append non-empty lines to teacher_runs.jsonl
  3. Count lines in both files
"""

import subprocess
import sys
from pathlib import Path
import json


def run_step_1():
    """Step 1: Generate teacher runs with llama3.1:70b"""
    print("\n" + "=" * 80)
    print("STEP 1: Generating teacher runs with llama3.1:70b")
    print("=" * 80)
    print("This may take a LONG time (several hours). Please wait...")
    print()

    cmd = [
        sys.executable,
        "scripts/generate_teacher_runs.py",
        "--all-tickets",
        "--tickets-file", "artifacts/generated_tickets_more.json",
        "--model", "llama3.1:70b",
        "--require-ollama-plan",
        "--require-ollama-draft",
        "--output-jsonl", "artifacts/teacher_runs_more.jsonl"
    ]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nERROR: Step 1 failed with exit code {result.returncode}")
        return False
    
    print("\n" + "=" * 80)
    print("STEP 1 COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return True


def run_step_2():
    """Step 2: Append non-empty lines to teacher_runs.jsonl"""
    print("\n" + "=" * 80)
    print("STEP 2: Appending non-empty lines to teacher_runs.jsonl")
    print("=" * 80)
    
    try:
        # Read non-empty lines from teacher_runs_more.jsonl
        more_path = Path("artifacts/teacher_runs_more.jsonl")
        if not more_path.exists():
            print(f"ERROR: {more_path} not found")
            return False
        
        more_lines = []
        with open(more_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    more_lines.append(line)
        
        # Append to teacher_runs.jsonl
        teacher_runs_path = Path("artifacts/teacher_runs.jsonl")
        with open(teacher_runs_path, 'a') as f:
            for line in more_lines:
                f.write(line + '\n')
        
        print(f"Appended {len(more_lines)} non-empty lines from teacher_runs_more.jsonl")
        print("=" * 80)
        print("STEP 2 COMPLETED SUCCESSFULLY")
        print("=" * 80)
        return True
    
    except Exception as e:
        print(f"ERROR in Step 2: {e}")
        return False


def run_step_3():
    """Step 3: Count lines in both files"""
    print("\n" + "=" * 80)
    print("STEP 3: Counting lines in both files")
    print("=" * 80)
    
    files = {
        'teacher_runs_more.jsonl': Path('artifacts/teacher_runs_more.jsonl'),
        'teacher_runs.jsonl': Path('artifacts/teacher_runs.jsonl')
    }
    
    for name, path in files.items():
        if path.exists():
            with open(path, 'r') as f:
                lines = f.readlines()
            non_empty = sum(1 for line in lines if line.strip())
            total = len(lines)
            size = path.stat().st_size
            
            print(f"\n{name}:")
            print(f"  Total lines: {total}")
            print(f"  Non-empty lines: {non_empty}")
            print(f"  File size: {size:,} bytes")
        else:
            print(f"\n{name}: FILE NOT FOUND")
    
    print("\n" + "=" * 80)
    print("STEP 3 COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return True


def main():
    """Run the complete pipeline"""
    print("\n")
    print("*" * 80)
    print("* TEACHER RUN PIPELINE")
    print("*" * 80)
    
    try:
        if not run_step_1():
            return 1
        
        if not run_step_2():
            return 1
        
        if not run_step_3():
            return 1
        
        print("\n")
        print("*" * 80)
        print("* PIPELINE COMPLETED SUCCESSFULLY!")
        print("*" * 80)
        print("\n")
        return 0
    
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
