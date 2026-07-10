#!/usr/bin/env python3
"""Measure how often SPARQLGraphBuilder.build_graph falls back to the
linear-chain-of-tokens representation (i.e. its triple-pattern regex finds
no matches), over the SQuARE anchors and candidates (positives + negatives)
for each knowledge graph.

Fills the placeholder in Sections/approach-v2.tex \S3.2 ("Fallback").

Usage:
    python scripts/measure_fallback_fraction.py
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SPARQLCE_DIR = ROOT_DIR / "sparqlce"


def _load_module_standalone(name, path):
    """Load a sparqlce submodule without importing sparqlce/__init__.py,
    which pulls in torch/transformers that this measurement doesn't need."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("sparqlce", types.ModuleType("sparqlce"))
_load_module_standalone("sparqlce.sparql_utils", SPARQLCE_DIR / "sparql_utils.py")
SPARQLGraphBuilder = _load_module_standalone(
    "sparqlce.graph_builder", SPARQLCE_DIR / "graph_builder.py"
).SPARQLGraphBuilder

DATASETS_DIR = ROOT_DIR / "datasets"

KG_FILES = {
    "dbpedia": [
        DATASETS_DIR / "dbpedia-triplets-dataset.json",
        DATASETS_DIR / "dbpedia_testset_RQ1.json",
    ],
    "wikidata": [
        DATASETS_DIR / "wikidata-triplets-dataset.json",
        DATASETS_DIR / "wikidata_testset_RQ1.json",
    ],
}


def collect_queries(paths):
    """Pull the anchor + positive + negative query strings out of the
    anchor/candidate triplet files, deduplicated."""
    queries = set()
    for path in paths:
        if not path.exists():
            print(f"  [skip] {path} not found", file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if "anchor_query" in item:
                queries.add(item["anchor_query"])
            for key in ("positive_queries", "negative_queries"):
                for entry in item.get(key, []):
                    q = entry.get("query") if isinstance(entry, dict) else entry
                    if q:
                        queries.add(q)
    return queries


def main():
    builder = SPARQLGraphBuilder()
    overall_total = 0
    overall_fallback = 0

    for kg, paths in KG_FILES.items():
        print(f"{kg}:")
        queries = collect_queries(paths)
        total = len(queries)
        fallback = 0
        for q in queries:
            builder.build_graph(q)
            if builder.last_used_fallback:
                fallback += 1

        pct = 100.0 * fallback / total if total else 0.0
        print(f"  unique queries: {total}")
        print(f"  fallback-path queries: {fallback}")
        print(f"  fallback fraction: {pct:.3f}%")

        overall_total += total
        overall_fallback += fallback

    overall_pct = 100.0 * overall_fallback / overall_total if overall_total else 0.0
    print("overall:")
    print(f"  unique queries: {overall_total}")
    print(f"  fallback-path queries: {overall_fallback}")
    print(f"  fallback fraction: {overall_pct:.3f}%")


if __name__ == "__main__":
    main()
