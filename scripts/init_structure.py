import os

base = r"C:\Users\harsh\.gemini\antigravity\scratch\solarscope-energy-analytics"
dirs = [
    os.path.join(base, "src", "data"),
    os.path.join(base, "src", "analytics"),
    os.path.join(base, "src", "models"),
    os.path.join(base, "src", "utils"),
    os.path.join(base, "app", "components"),
    os.path.join(base, "app", "views"),
    os.path.join(base, "tests"),
    os.path.join(base, "docs", "screenshots"),
    os.path.join(base, "data", "processed"),
    os.path.join(base, ".github", "workflows"),
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

# create __init__.py files
init_paths = [
    os.path.join(base, "src", "__init__.py"),
    os.path.join(base, "src", "data", "__init__.py"),
    os.path.join(base, "src", "analytics", "__init__.py"),
    os.path.join(base, "src", "models", "__init__.py"),
    os.path.join(base, "src", "utils", "__init__.py"),
    os.path.join(base, "app", "__init__.py"),
    os.path.join(base, "app", "components", "__init__.py"),
    os.path.join(base, "app", "views", "__init__.py"),
    os.path.join(base, "tests", "__init__.py"),
]

for p in init_paths:
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            f.write('"""SolarScope package initialization."""\n')

print("All directories and __init__.py files created.")
