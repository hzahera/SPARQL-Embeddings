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
import urllib.error
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


def _request_chunk(chunk: list[str], max_retries: int = 6) -> dict:
    """Fetch one <=50-id chunk with exponential backoff on 429/5xx."""
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
        f"{API}?{params}",
        headers={"User-Agent": "SPARQLCE-label-builder/1.0 (research; contact via repo)"},
    )
    delay = 2.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                wait = float(e.headers.get("Retry-After", delay))
                print(f"  HTTP {e.code}; backing off {wait:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                time.sleep(wait)
                delay = min(delay * 2, 120)  # exponential, capped at 2 min
                continue
            raise
    raise RuntimeError("exhausted retries")


def save(labels: dict[str, str]) -> None:
    json.dump(labels, open(OUT, "w"), ensure_ascii=False, indent=0)


def fetch_labels(ids: list[str]) -> dict[str, str]:
    # Resume: keep labels already resolved in a previous (interrupted) run.
    labels: dict[str, str] = {}
    if OUT.exists():
        try:
            labels = json.load(open(OUT))
            print(f"Resuming: {len(labels)} labels already on disk", file=sys.stderr)
        except ValueError:
            pass

    todo = [i for i in ids if i not in labels]
    print(f"{len(todo)} ids left to resolve", file=sys.stderr)

    for i in range(0, len(todo), 50):  # API caps wbgetentities at 50 ids/call
        chunk = todo[i : i + 50]
        payload = _request_chunk(chunk)
        for eid, ent in payload.get("entities", {}).items():
            lab = ent.get("labels", {}).get("en", {}).get("value")
            # Record even a missing label as "" so we don't re-query it on resume.
            labels[eid] = lab if lab else labels.get(eid, "")
        save(labels)  # checkpoint after every chunk so 429s never lose progress
        print(f"  resolved {i + len(chunk)}/{len(todo)}", file=sys.stderr)
        time.sleep(0.5)  # gentler default cadence to avoid rate limits

    return labels


def main() -> None:
    ids = collect_ids(DATA)
    print(f"Found {len(ids)} unique Wikidata ids in {DATA.name}", file=sys.stderr)
    labels = fetch_labels(ids)
    save(labels)
    resolved = sum(1 for v in labels.values() if v)
    print(f"Wrote {len(labels)} entries to {OUT} ({resolved} with an en label)")


if __name__ == "__main__":
    main()
