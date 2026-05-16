#!/usr/bin/env python3
"""Verify that the SPARQLCE graph builder bakes in two invariances:

  A. Variable renaming: ``?person`` ↔ ``?x`` produces identical graphs and
     therefore identical embeddings.
  B. FILTER(true): a trivially-true filter is a no-op and is skipped during
     graph construction, so adding one does not change the embedding.

The graph-level checks (Part 1) are independent of any trained checkpoint and
should pass purely from the code edits in ``sparqlce/graph_builder.py`` and
``sparqlce/sparql_utils.py``.

The embedding-level checks (Part 2) additionally require a SPARQLCE
checkpoint. Pass one with ``--ckpt path/to/triplet_model.pt`` to run them.

Usage:
    python scripts/verify_invariances.py
    python scripts/verify_invariances.py --ckpt checkpoints/triplet_model_semantic_canon_v2-20.pt
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from sparqlce.graph_builder import SPARQLGraphBuilder


# ----------------------------- test pairs -----------------------------
RENAME_PAIRS = [
    (
        "SELECT ?person WHERE { ?person <http://dbpedia.org/ontology/birthPlace> ?place }",
        "SELECT ?x      WHERE { ?x      <http://dbpedia.org/ontology/birthPlace> ?y     }",
    ),
    (
        "SELECT ?a ?b WHERE { ?a <p> ?b . ?b <q> ?a }",
        "SELECT ?u ?v WHERE { ?u <p> ?v . ?v <q> ?u }",
    ),
]

FILTER_TRUE_PAIRS = [
    (
        "SELECT ?x WHERE { ?x <p> ?y }",
        "SELECT ?x WHERE { ?x <p> ?y . FILTER(true) }",
    ),
    (
        "SELECT ?x WHERE { ?x <p> ?y }",
        "SELECT ?x WHERE { ?x <p> ?y . FILTER ( TRUE ) }",
    ),
]


def graph_signature(G):
    """A canonical, comparable representation of a labelled DiGraph."""
    nodes = tuple(sorted((n, G.nodes[n]["label"]) for n in G.nodes))
    edges = tuple(sorted((u, v, d.get("label", "")) for u, v, d in G.edges(data=True)))
    return nodes, edges


def check_graph_invariance(builder, pairs, label):
    print(f"\n[Graph-level] {label}")
    all_ok = True
    for q1, q2 in pairs:
        s1 = graph_signature(builder.build_graph(q1))
        s2 = graph_signature(builder.build_graph(q2))
        ok = s1 == s2
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {q1[:50]}...  vs  {q2[:50]}...")
        if not ok:
            print(f"    nodes A: {s1[0]}")
            print(f"    nodes B: {s2[0]}")
            print(f"    edges A: {s1[1]}")
            print(f"    edges B: {s2[1]}")
        all_ok &= ok
    return all_ok


def check_embedding_invariance(embedder, pairs, label):
    import numpy as np

    print(f"\n[Embedding-level] {label}")
    all_ok = True
    for q1, q2 in pairs:
        v1, v2 = embedder.encode([q1, q2])
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        ok = cos > 0.9999
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] cos = {cos:.6f}    {q1[:40]}...  vs  {q2[:40]}...")
        all_ok &= ok
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="Optional SPARQLCE checkpoint for embedding-level checks")
    args = parser.parse_args()

    builder = SPARQLGraphBuilder()

    g_ok_rename = check_graph_invariance(builder, RENAME_PAIRS, "variable renaming")
    g_ok_filter = check_graph_invariance(builder, FILTER_TRUE_PAIRS, "FILTER(true) no-op")

    e_ok_rename = True
    e_ok_filter = True
    if args.ckpt:
        from sparqlce.embedder import SPARQLEmbedder

        embedder = SPARQLEmbedder(args.ckpt)
        e_ok_rename = check_embedding_invariance(embedder, RENAME_PAIRS, "variable renaming")
        e_ok_filter = check_embedding_invariance(embedder, FILTER_TRUE_PAIRS, "FILTER(true) no-op")
    else:
        print("\n(skipping embedding-level checks — pass --ckpt to enable)")

    print("\n========== summary ==========")
    print(f"  graph rename    : {'PASS' if g_ok_rename else 'FAIL'}")
    print(f"  graph filter    : {'PASS' if g_ok_filter else 'FAIL'}")
    if args.ckpt:
        print(f"  embed rename    : {'PASS' if e_ok_rename else 'FAIL'}")
        print(f"  embed filter    : {'PASS' if e_ok_filter else 'FAIL'}")

    overall = all([g_ok_rename, g_ok_filter, e_ok_rename, e_ok_filter])
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
