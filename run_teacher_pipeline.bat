@echo off
REM Teacher Run Pipeline - Step 1, 2, 3
REM This script runs the complete teacher generation pipeline

cd /d "C:\Users\parav\Documents\New project 2"

echo.
echo ============================================================
echo STEP 1: Generating teacher runs with llama3.1:70b
echo ============================================================
echo This may take a LONG time (several hours). Please wait...
echo.

.venv\Scripts\python.exe scripts/generate_teacher_runs.py ^
    --all-tickets ^
    --tickets-file artifacts/generated_tickets_more.json ^
    --model llama3.1:70b ^
    --require-ollama-plan ^
    --require-ollama-draft ^
    --output-jsonl artifacts/teacher_runs_more.jsonl

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Step 1 failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo STEP 2: Appending non-empty lines to teacher_runs.jsonl
echo ============================================================

.venv\Scripts\python.exe -c "
import json

# Read non-empty lines from teacher_runs_more.jsonl
more_lines = []
try:
    with open('artifacts/teacher_runs_more.jsonl', 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                more_lines.append(line)
except FileNotFoundError:
    print('ERROR: artifacts/teacher_runs_more.jsonl not found')
    exit(1)

# Append to teacher_runs.jsonl
with open('artifacts/teacher_runs.jsonl', 'a') as f:
    for line in more_lines:
        f.write(line + '\n')

print(f'Appended {len(more_lines)} lines from teacher_runs_more.jsonl')
"

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Step 2 failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo STEP 3: Counting lines in both files
echo ============================================================

.venv\Scripts\python.exe -c "
from pathlib import Path

files = {
    'teacher_runs_more.jsonl': 'artifacts/teacher_runs_more.jsonl',
    'teacher_runs.jsonl': 'artifacts/teacher_runs.jsonl'
}

for name, path in files.items():
    p = Path(path)
    if p.exists():
        with open(p, 'r') as f:
            lines = f.readlines()
        non_empty = sum(1 for line in lines if line.strip())
        total = len(lines)
        print(f'{name}:')
        print(f'  Total lines: {total}')
        print(f'  Non-empty lines: {non_empty}')
        print(f'  File size: {p.stat().st_size} bytes')
    else:
        print(f'{name}: FILE NOT FOUND')
    print()

print('Pipeline completed successfully!')
"

echo.
echo ============================================================
echo PIPELINE COMPLETED
echo ============================================================
echo.
pause
