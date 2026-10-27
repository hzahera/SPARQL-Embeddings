#!/usr/bin/env python3
"""Bundle each demo page (index, embeddings, tasks) with its CSS, JS and data into ONE self-contained file.

The results (docs/*.html) can be served by GitHub Pages ("Deploy from a branch",
folder /docs) or by the "GitHub Pages" option of Anonymous GitHub, and it also works
when opened directly from disk.

Usage (from the repository root):
    python demo/scripts/bundle_demo.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo"
OUT = ROOT / "docs"


def main():
    OUT.mkdir(exist_ok=True)
    for page in ("index.html", "embeddings.html", "tasks.html"):
        build(page)
    (OUT / ".nojekyll").write_text("", encoding="utf-8")


def build(page):
    html = (DEMO / page).read_text(encoding="utf-8")

    def css(m):
        return "<style>\n" + (DEMO / m.group(1)).read_text(encoding="utf-8") + "\n</style>"

    def js(m):
        code = (DEMO / m.group(1)).read_text(encoding="utf-8").replace("</script", "<\\/script")
        return "<script>\n" + code + "\n</script>"

    html = re.sub(r'<link rel="stylesheet" href="([^"]+)">', css, html)
    html = re.sub(r'<script src="([^"]+)"></script>', js, html)
    (OUT / page).write_text(html, encoding="utf-8")
    print("wrote", OUT / page, f"{len(html) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
