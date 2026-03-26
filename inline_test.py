import subprocess
from pathlib import Path

# Check current state
more_file = Path(r"C:\Users\parav\Documents\New project 2\artifacts\teacher_runs_more.jsonl")
total_file = Path(r"C:\Users\parav\Documents\New project 2\artifacts\teacher_runs.jsonl")

def count_lines(fp):
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            return len([l for l in f if l.strip()])
    except:
        return 0

more_cnt = count_lines(more_file)
total_cnt = count_lines(total_file)

print(f"Current: teacher_runs_more.jsonl={more_cnt}, teacher_runs.jsonl={total_cnt}")

if more_cnt == 0:
    print("\nGenerating pipeline data...")
    repo = Path(r"C:\Users\parav\Documents\New project 2")
    py = repo / ".venv\Scripts\python.exe"
    
    print("Step 1: synthetic tickets...", end=" ", flush=True)
    r1 = subprocess.run([str(py), "scripts/generate_synthetic_tickets.py", 
                        "--access-count", "1", "--billing-count", "1",
                        "--integration-count", "1", "--escalation-count", "1",
                        "--output", "artifacts/generated_tickets_more.json"],
                       cwd=str(repo), capture_output=True, text=True)
    print("OK" if r1.returncode == 0 else f"FAIL({r1.returncode})")
    if r1.returncode != 0:
        print(r1.stderr)
    
    print("Step 2: teacher runs with llama3.1:70b...", end=" ", flush=True)
    r2 = subprocess.run([str(py), "scripts/generate_teacher_runs.py",
                        "--all-tickets", 
                        "--tickets-file", "artifacts/generated_tickets_more.json",
                        "--model", "llama3.1:70b",
                        "--require-ollama-plan", "--require-ollama-draft",
                        "--output-jsonl", "artifacts/teacher_runs_more.jsonl"],
                       cwd=str(repo), capture_output=True, text=True, timeout=600)
    print("OK" if r2.returncode == 0 else f"FAIL({r2.returncode})")
    if r2.returncode != 0:
        print("STDERR:", r2.stderr[-500:] if len(r2.stderr) > 500 else r2.stderr)
    print("STDOUT:", r2.stdout[-200:] if len(r2.stdout) > 200 else r2.stdout)
    
    more_cnt = count_lines(more_file)
    
    print(f"Step 3: appending {more_cnt} lines...")
    if more_cnt > 0:
        with open(more_file, 'r') as src:
            lines = [l for l in src if l.strip()]
        with open(total_file, 'a') as dst:
            for line in lines:
                dst.write(line if line.endswith('\n') else line + '\n')
    
    more_cnt = count_lines(more_file)
    total_cnt = count_lines(total_file)

print(f"\n{'='*60}")
print(f"FINAL COUNTS:")
print(f"  teacher_runs_more.jsonl: {more_cnt} lines")
print(f"  teacher_runs.jsonl: {total_cnt} lines")
