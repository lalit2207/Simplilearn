import re
from pathlib import Path

js_dir = Path(".venv/Lib/site-packages/streamlit/static/static/js")
path = next(js_dir.glob("index*.js"))
text = path.read_text(encoding="utf-8", errors="ignore")
ids = sorted(set(re.findall(r"st[A-Z][A-Za-z0-9]+", text)))
for item in ids:
    low = item.lower()
    if any(word in low for word in ("sidebar", "collaps", "expand", "nav")):
        print(item)
