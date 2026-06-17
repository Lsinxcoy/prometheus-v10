"""Batch rename prometheus_v10 -> prometheus_v10 in all Python files."""
import os

count = 0
for root, dirs, files in os.walk("src"):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            new_content = content.replace("prometheus_v10", "prometheus_v10")
            if new_content != content:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(new_content)
                count += 1

# Also update root-level test files
for f in os.listdir("."):
    if f.endswith(".py"):
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
        new_content = content.replace("prometheus_v10", "prometheus_v10")
        if new_content != content:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(new_content)
            count += 1

# Update pyproject.toml
for f in ["pyproject.toml"]:
    if os.path.exists(f):
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
        new_content = content.replace("prometheus_v10", "prometheus_v10").replace("prometheus-v9pro", "prometheus-v10")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(new_content)
        count += 1

print(f"Updated {count} files")
