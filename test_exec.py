import subprocess
import sys
from pathlib import Path

repo_root = Path(r"C:\Users\parav\Documents\New project 2")
python_exe = repo_root / ".venv" / "Scripts" / "python.exe"

# Step 1
print("Step 1: Generate synthetic tickets...")
r = subprocess.run([str(python_exe), "scripts/generate_synthetic_tickets.py",
                    "--access-count", "1", "--billing-count", "1",
                    "--integration-count", "1", "--escalation-count", "1",
                    "--output", "artifacts/generated_tickets_more.json"],
                   cwd=str(repo_root), capture_output=True, text=True, timeout=60)
print(r.stdout if r.returncode == 0 else f"ERROR: {r.stderr}")
if r.returncode != 0:
    sys.exit(1)

# Step 2
print("\nStep 2: Generate teacher runs (llama3.1:70b - this will take time)...")
r = subprocess.run([str(python_exe), "scripts/generate_teacher_runs.py",
                    "--all-tickets", "--tickets-file", "artifacts/generated_tickets_more.json",
                    "--model", "llama3.1:70b", "--require-ollama-plan", "--require-ollama-draft",
                    "--output-jsonl", "artifacts/teacher_runs_more.jsonl"],
                   cwd=str(repo_root), timeout=1800)  # 30 min timeout
if r.returncode != 0:
    sys.exit(1)

# Step 3
print("\nStep 3: Appending non-empty lines...")
more_file = repo_root / "artifacts" / "teacher_runs_more.jsonl"
total_file = repo_root / "artifacts" / "teacher_runs.jsonl"

if more_file.exists():
    with open(more_file, 'r') as f:
        lines = [l for l in f if l.strip()]
    with open(total_file, 'a') as f:
        for line in lines:
            f.write(line if line.endswith('\n') else line + '\n')
    print(f"Appended {len(lines)} lines")

# Step 4 - Final counts
def count_lines(fp):
    try:
        with open(fp) as f:
            return len([l for l in f if l.strip()])
    except:
        return 0

print(f"\nFinal Counts:")
print(f"  artifacts/teacher_runs_more.jsonl: {count_lines(more_file)} lines")
print(f"  artifacts/teacher_runs.jsonl: {count_lines(total_file)} lines")
