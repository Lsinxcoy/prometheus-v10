import os

base = "E:/prometheus-v9pro/src/prometheus_v10"
total = 0
by_dir = {}
file_count = 0
for root, dirs, files in os.walk(base):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as fh:
                lines = sum(1 for _ in fh)
            total += lines
            file_count += 1
            rel = os.path.relpath(root, base)
            by_dir[rel] = by_dir.get(rel, 0) + lines

print(f"Total: {file_count} files, {total} lines")
for d, lines in sorted(by_dir.items(), key=lambda x: -x[1]):
    print(f"  {d}: {lines} lines")
