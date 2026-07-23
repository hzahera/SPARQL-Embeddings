#!/usr/bin/env python3
"""Resolve every Wikidata P/Q id used in the training corpus to its English label.

Why: Wikidata predicates/entities are opaque ids (wdt:P31, wde:Q5). The E5 text
backbone in HybridSPARQLEncoder embeds *label strings*, so on Wikidata its text
stream sees "P31"/"Q5" — semantically empty — and the gated fusion learns to
ignore it. Expanding P31 -> "instance of", Q5 -> "human" turns those ids back
into the kind of prose E5 already exploits on DBpedia, reviving the dead stream.

Run once; the graph builder picks up datasets/wikidata_labels.json automatically.

    python scripts/build_wikidata_labels.py
    python scripts/build_wikidata_labels.py --endpoint https://www.wikidata.org/w/api.php --api

By default this queries the project's DICE Wikidata mirror over SPARQL, which is
not rate-limited (unlike the public REST API) and resolves all ~4.2k ids in ~14
batched queries. Progress is checkpointed to disk after every batch, so an
interrupted run resumes where it left off.
"""
import argparse
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

# Project's self-hosted Wikidata mirror (same host family as the DBpedia endpoint
# used in notebooks/evaluation-RQ1/DBpedia/07_visualize_alignment_dbpedia.ipynb).
SPARQL_ENDPOINT = "https://wikidata.data.dice-research.org/sparql"
REST_API = "https://www.wikidata.org/w/api.php"

ID_RE = re.compile(r"\b[PQ]\d+\b")
# Wikidata property/entity URIs both live under the entity namespace, e.g.
# <http://www.wikidata.org/entity/P31> and <.../Q5>, and both carry rdfs:label.
_ENTITY_NS = "http://www.wikidata.org/entity/"
_LABEL_QUERY = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?e ?label WHERE {{
  VALUES ?e {{ {values} }}
  ?e rdfs:label ?label .
  FILTER (lang(?label) = "en")
}}"""


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


def _http_get(url: str, accept: str, max_retries: int = 6) -> dict:
    """GET with exponential backoff on 429/5xx; returns parsed JSON."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "SPARQLCE-label-builder/1.0 (research; contact via repo)",
            "Accept": accept,
        },
    )
    delay = 2.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                wait = float(e.headers.get("Retry-After", delay))
                print(f"  HTTP {e.code}; backing off {wait:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                time.sleep(wait)
                delay = min(delay * 2, 120)
                continue
            raise
    raise RuntimeError("exhausted retries")


def _sparql_batch(endpoint: str, chunk: list[str]) -> dict[str, str]:
    """Resolve one batch of ids via a single VALUES SPARQL query."""
    values = " ".join(f"wd:{i}" for i in chunk)
    query = _LABEL_QUERY.format(values=values)
    url = endpoint + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    payload = _http_get(url, accept="application/sparql-results+json")
    out: dict[str, str] = {}
    for b in payload.get("results", {}).get("bindings", []):
        uri = b["e"]["value"]
        eid = uri[len(_ENTITY_NS):] if uri.startswith(_ENTITY_NS) else uri
        out[eid] = b["label"]["value"]
    return out


def _rest_batch(endpoint: str, chunk: list[str]) -> dict[str, str]:
    """Resolve one <=50-id batch via the wbgetentities REST API (fallback)."""
    url = endpoint + "?" + urllib.parse.urlencode(
        {
            "action": "wbgetentities",
            "ids": "|".join(chunk),
            "props": "labels",
            "languages": "en",
            "format": "json",
        }
    )
    payload = _http_get(url, accept="application/json")
    out: dict[str, str] = {}
    for eid, ent in payload.get("entities", {}).items():
        lab = ent.get("labels", {}).get("en", {}).get("value")
        if lab:
            out[eid] = lab
    return out


def save(labels: dict[str, str]) -> None:
    json.dump(labels, open(OUT, "w"), ensure_ascii=False, indent=0)


def fetch_labels(ids: list[str], endpoint: str, use_api: bool, batch: int, pause: float) -> dict[str, str]:
    # Resume: keep labels already resolved in a previous (interrupted) run.
    labels: dict[str, str] = {}
    if OUT.exists():
        try:
            labels = json.load(open(OUT))
            print(f"Resuming: {len(labels)} entries already on disk", file=sys.stderr)
        except ValueError:
            pass

    todo = [i for i in ids if i not in labels]
    print(f"{len(todo)} ids left to resolve via {'REST API' if use_api else 'SPARQL'} "
          f"({endpoint})", file=sys.stderr)

    fetch_batch = _rest_batch if use_api else _sparql_batch
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        resolved = fetch_batch(endpoint, chunk)
        for eid in chunk:
            # Record "" for ids the endpoint returned no label for, so a resume
            # doesn't re-query them every run.
            labels[eid] = resolved.get(eid, labels.get(eid, ""))
        save(labels)  # checkpoint after every batch
        got = sum(1 for eid in chunk if labels.get(eid))
        print(f"  {min(i + batch, len(todo))}/{len(todo)} ids  (+{got} labels this batch)",
              file=sys.stderr)
        if pause:
            time.sleep(pause)

    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=None,
                    help="override endpoint URL (default: DICE SPARQL mirror, or "
                         "the public REST API when --api is set)")
    ap.add_argument("--api", action="store_true",
                    help="use the public wbgetentities REST API instead of SPARQL")
    ap.add_argument("--batch", type=int, default=None,
                    help="ids per request (default: 300 for SPARQL, 50 for REST)")
    ap.add_argument("--pause", type=float, default=0.0,
                    help="seconds to sleep between batches")
    args = ap.parse_args()

    endpoint = args.endpoint or (REST_API if args.api else SPARQL_ENDPOINT)
    batch = args.batch or (50 if args.api else 300)

    ids = collect_ids(DATA)
    print(f"Found {len(ids)} unique Wikidata ids in {DATA.name}", file=sys.stderr)
    labels = fetch_labels(ids, endpoint, args.api, batch, args.pause)
    save(labels)
    resolved = sum(1 for v in labels.values() if v)
    print(f"Wrote {len(labels)} entries to {OUT} ({resolved} with an en label)")


if __name__ == "__main__":
    main()
