from pathlib import Path

artifacts = Path(r"C:\Users\parav\Documents\New project 2\artifacts")

for f in artifacts.glob("*.jsonl"):
    size = f.stat().st_size
    with open(f) as fh:
        lines = len([l for l in fh if l.strip()])
    print(f"{f.name}: {size} bytes, {lines} non-empty lines")
