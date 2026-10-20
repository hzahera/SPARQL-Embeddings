#!/usr/bin/env python3
"""Build the matched syntax-label training split for the syntax-supervised control.

Only the labeling rule changes. For every training anchor of SQuARE:
  * positives: the |P_a| training queries with the highest algebra similarity to the
    anchor, where |P_a| is the anchor's number of answer-overlap positives;
  * negatives: |N_a| training queries from the anchor's low-similarity zone
    (similarity <= neg_ratio * similarity of its weakest syntax positive). The first
    n_close of them are the most similar queries in that zone, where n_close is the
    anchor's number of nonzero-overlap answer negatives (0 < J <= 0.1); the rest are
    drawn at random from the zone. This mirrors how SQuARE builds its negatives.
Anchors, splits, counts, loss and hyperparameters stay identical. Validation and test
files are NOT touched: checkpoint selection and evaluation keep using answer overlap.

Algebra similarity is the same multiset Jaccard as the `algebra_similarity` baseline
(scripts/baselines/nonneural.py): algebra node labels, parent>child edges,
variable-abstracted triple patterns and RDF terms.

Candidates carry no `answer_jaccard` field, so the data loader assigns 1.0/0.0 and the
runtime validator passes. `syntax_similarity` and, where the pair is stored in SQuARE,
`stored_answer_jaccard` are kept for auditing only.

Usage (from the repository root):
  PYTHONHASHSEED=0 python scripts/syntax_control/build_syntax_labels.py --graph dbpedia
Outputs:
  datasets/<graph>-splits/<graph>-train-syntax.json
  scripts/syntax_control/label_stats_<graph>.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "audit"))
import audit_splits as au  # noqa: E402  (ws, add_prefixes)

from rdflib import BNode, Literal, URIRef, Variable  # noqa: E402
from rdflib.plugins.sparql.algebra import translateQuery  # noqa: E402
from rdflib.plugins.sparql.parser import parseQuery  # noqa: E402
from rdflib.plugins.sparql.parserutils import CompValue  # noqa: E402

ws = au.ws


TOK = re.compile(r"<[^>]*>|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|[?$]\w+|[\w\-]*:[\w\-./#]*|\w+|[^\s\w]")


def token_bag(q: str) -> Counter:
    """Fallback for queries rdflib cannot parse: multiset of SPARQL tokens
    (same tokenizer as the TF-IDF baseline, variables masked)."""
    return Counter("tok:" + ("?var" if t[0] in "?$" else t) for t in TOK.findall(q))


def struct_bag(q: str):
    """Same features as nonneural.struct_bag; token bag if rdflib cannot parse."""
    try:
        alg = translateQuery(parseQuery(au.add_prefixes(q))).algebra
    except Exception:
        return ("fallback", token_bag(q))
    bag: Counter = Counter()

    def lab(x):
        if isinstance(x, CompValue):
            return x.name
        if isinstance(x, Variable):
            return "?v"
        if isinstance(x, BNode):
            return "_:b"
        if isinstance(x, (URIRef, Literal)):
            return str(x)
        return None

    def walk(node, parent):
        if isinstance(node, CompValue):
            l = node.name
            bag[l] += 1
            if parent:
                bag[parent + ">" + l] += 1
            for k, v in node.items():
                if k.startswith("_") or k in ("PV", "datasetClause"):
                    continue
                walk(v, l)
        elif isinstance(node, (list, tuple)):
            if len(node) == 3 and all(isinstance(t, (URIRef, Literal, Variable, BNode)) for t in node):
                bag["TP(" + ",".join(lab(t) for t in node) + ")"] += 1
                if parent:
                    bag[parent + ">TP"] += 1
                for t in node:
                    bag["term:" + lab(t)] += 1
            else:
                for v in node:
                    walk(v, parent)
        elif lab(node) is not None and parent:
            bag[parent + ">" + lab(node)] += 1

    walk(alg, None)
    return bag


def bag_matrix(bags):
    """Expand each multiset into occurrence tokens f#1..f#c, so that multiset Jaccard
    equals set Jaccard and the intersection is a sparse dot product."""
    vocab: dict[str, int] = {}
    rows, cols = [], []
    for i, bag in enumerate(bags):
        for f, c in (bag or {}).items():
            for k in range(1, c + 1):
                j = vocab.setdefault(f"{f}#{k}", len(vocab))
                rows.append(i)
                cols.append(j)
    X = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(bags), max(len(vocab), 1))
    )
    return X, np.asarray(X.sum(axis=1)).ravel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, choices=["dbpedia", "wikidata"])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--neg-ratio", type=float, default=0.5,
                    help="negative zone: similarity <= ratio * weakest syntax positive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.root)
    d = root / "datasets" / f"{args.graph}-splits"
    train = json.load(open(d / f"{args.graph}-train.json", encoding="utf-8"))

    # Training query universe, as in nonneural.py.
    queries = list(dict.fromkeys(
        ws(q) for r in train
        for q in [r["anchor_query"]] + [x["query"] for x in r["positive_queries"] + r["negative_queries"]]
    ))
    idx = {q: i for i, q in enumerate(queries)}
    surface: dict[str, str] = {}
    for r in train:
        for q in [r["anchor_query"]] + [x["query"] for x in r["positive_queries"] + r["negative_queries"]]:
            surface.setdefault(ws(q), q)

    print(f"{args.graph}: {len(train)} anchors, {len(queries)} training queries; parsing...", flush=True)
    with Pool(args.workers) as p:
        bags = p.map(struct_bag, queries, chunksize=20)
    parse_failures = sum(isinstance(b, tuple) for b in bags)
    bags = [b[1] if isinstance(b, tuple) else b for b in bags]
    X, size = bag_matrix(bags)
    XT = X.T.tocsc()

    out, stats_rows = [], []
    for ai, r in enumerate(train):
        a = idx[ws(r["anchor_query"])]
        n_pos = len(r["positive_queries"])
        n_neg = len(r["negative_queries"])
        n_close = sum(1 for x in r["negative_queries"] if x["answer_jaccard"] > 0)
        stored = {ws(x["query"]): x["answer_jaccard"] for x in r["positive_queries"] + r["negative_queries"]}

        inter = np.asarray((X[a] @ XT).todense()).ravel()
        union = size[a] + size - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        sim[a] = -1.0  # never pick the anchor itself

        rng = random.Random(args.seed * 1_000_003 + ai)
        tie = np.array([rng.random() for _ in range(len(queries))])
        order = np.lexsort((tie, -sim))  # similarity desc, random tie-break
        order = order[order != a]

        pos = order[:n_pos]
        s_min = float(sim[pos[-1]])
        zone = [j for j in order[n_pos:] if sim[j] <= args.neg_ratio * s_min]
        fallback = len(zone) < n_neg
        if fallback:  # too few low-similarity queries: take the least similar ones
            zone = list(order[::-1][: max(n_neg, 1) * 3])
            zone = [j for j in zone if j not in set(pos)]
            zone.sort(key=lambda j: (-sim[j], tie[j]))
        close = list(zone[:n_close])
        rest = [j for j in zone[n_close:]]
        neg = close + rng.sample(rest, n_neg - len(close))

        def cand(j):
            q = queries[j]
            c = {"query": surface[q], "syntax_similarity": round(float(sim[j]), 6)}
            if q in stored:
                c["stored_answer_jaccard"] = stored[q]
            return c

        out.append({
            "anchor_query": r["anchor_query"],
            "positive_queries": [cand(j) for j in pos],
            "negative_queries": [cand(j) for j in neg],
        })
        ans_pos = {ws(x["query"]) for x in r["positive_queries"]}
        stats_rows.append({
            "pos_sim_min": s_min,
            "pos_sim_mean": float(sim[pos].mean()),
            "neg_sim_max": float(max(sim[j] for j in neg)),
            "zone_size": len(zone),
            "fallback": fallback,
            "pos_overlap_with_answer_pos": len(ans_pos & {queries[j] for j in pos}) / n_pos,
        })

    dst = d / f"{args.graph}-train-syntax.json"
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def q(vals, p):
        return round(float(np.percentile(vals, p)), 4)

    S = {k: [row[k] for row in stats_rows] for k in stats_rows[0]}
    stats = {
        "graph": args.graph,
        "anchors": len(out),
        "training_queries": len(queries),
        "parse_failures": parse_failures,
        "neg_ratio": args.neg_ratio,
        "seed": args.seed,
        "positives": sum(len(r["positive_queries"]) for r in out),
        "negatives": sum(len(r["negative_queries"]) for r in out),
        "anchors_using_fallback_zone": int(sum(S["fallback"])),
        "positive_similarity_min_median": q(S["pos_sim_min"], 50),
        "positive_similarity_mean_median": q(S["pos_sim_mean"], 50),
        "negative_similarity_max_median": q(S["neg_sim_max"], 50),
        "share_of_syntax_positives_that_are_answer_positives": round(float(np.mean(S["pos_overlap_with_answer_pos"])), 4),
        "share_of_syntax_positive_pairs_with_stored_J_ge_0.8": round(float(np.mean([
            c.get("stored_answer_jaccard", -1) >= 0.8 for r in out for c in r["positive_queries"]])), 4),
        "share_of_syntax_negative_pairs_with_stored_J_ge_0.8": round(float(np.mean([
            c.get("stored_answer_jaccard", -1) >= 0.8 for r in out for c in r["negative_queries"]])), 4),
        "output": str(dst.relative_to(root)),
    }
    stats_path = root / "scripts" / "syntax_control" / f"label_stats_{args.graph}.json"
    json.dump(stats, open(stats_path, "w"), indent=2)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
