import os

base = "E:/prometheus-v9pro/src/prometheus_v10"
count = 0
for root, dirs, files in os.walk(base):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        if not f.endswith(".py"):
            continue
        path = os.path.join(root, f)
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        new_content = content.replace("prometheus_v9.", "prometheus_v10.").replace("from prometheus_v9 ", "from prometheus_v10 ").replace("import prometheus_v9", "import prometheus_v10")
        if new_content != content:
            count += 1
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(new_content)

# Also update pyproject.toml
toml_path = "E:/prometheus-v9pro/pyproject.toml"
with open(toml_path, "r", encoding="utf-8") as fh:
    content = fh.read()
new_content = content.replace("prometheus-v9", "prometheus-v9pro").replace("prometheus_v9", "prometheus_v10")
with open(toml_path, "w", encoding="utf-8") as fh:
    fh.write(new_content)

print(f"Updated {count} source files + pyproject.toml")
