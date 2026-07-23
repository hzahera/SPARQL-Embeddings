#!/usr/bin/env python3
"""Resolve every Wikidata P/Q id used in the training corpus to its English label.

Why: Wikidata predicates/entities are opaque ids (wdt:P31, wde:Q5). The E5 text
backbone in HybridSPARQLEncoder embeds *label strings*, so on Wikidata its text
stream sees "P31"/"Q5" — semantically empty — and the gated fusion learns to
ignore it. Expanding P31 -> "instance of", Q5 -> "human" turns those ids back
into the kind of prose E5 already exploits on DBpedia, reviving the dead stream.

Run once; the graph builder picks up datasets/wikidata_labels.json automatically.

    python scripts/build_wikidata_labels.py

Requires network access to the Wikidata API. ~4.2k ids -> ~85 API calls -> ~1 min.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "wikidata-triplets-dataset.json"
OUT = ROOT / "datasets" / "wikidata_labels.json"
API = "https://www.wikidata.org/w/api.php"
ID_RE = re.compile(r"\b[PQ]\d+\b")


def collect_ids(path: Path) -> list[str]:
    data = json.load(open(path))
    ids: set[str] = set()

    def scan(q: str) -> None:
        ids.update(ID_RE.findall(q))

    for g in data:
        scan(g["anchor_query"])
        for key in ("positive_queries", "negative_queries"):
            for x in g.get(key, []):
                scan(x["query"])
    return sorted(ids, key=lambda s: (s[0], int(s[1:])))


def fetch_labels(ids: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for i in range(0, len(ids), 50):  # API caps wbgetentities at 50 ids/call
        chunk = ids[i : i + 50]
        params = urllib.parse.urlencode(
            {
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": "labels",
                "languages": "en",
                "format": "json",
            }
        )
        req = urllib.request.Request(
            f"{API}?{params}", headers={"User-Agent": "SPARQLCE-label-builder/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.load(r)
        for eid, ent in payload.get("entities", {}).items():
            lab = ent.get("labels", {}).get("en", {}).get("value")
            if lab:
                labels[eid] = lab
        print(f"  resolved {i + len(chunk)}/{len(ids)}", file=sys.stderr)
        time.sleep(0.1)  # be polite to the public API
    return labels


def main() -> None:
    ids = collect_ids(DATA)
    print(f"Found {len(ids)} unique Wikidata ids in {DATA.name}", file=sys.stderr)
    labels = fetch_labels(ids)
    json.dump(labels, open(OUT, "w"), ensure_ascii=False, indent=0)
    missing = len(ids) - len(labels)
    print(f"Wrote {len(labels)} labels to {OUT} ({missing} ids had no en label)")


if __name__ == "__main__":
    main()
