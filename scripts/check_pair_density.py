#!/usr/bin/env python3
"""Does a candidate corpus contain enough result-equivalent pairs to train on?

Structural richness and pair density pull against each other, and the trade-off
decides whether a corpus is usable at all. SQuARE-DBpedia has 39,520 positive
pairs over 4,713 anchors *because* it is full of near-duplicate template
families -- the same property that leaves 49.1% of its queries without a join
vertex and makes its graphs diameter-1. Filter for structure and the pairs may
go with it.

This measures that before any oracle is rebuilt or any model is trained:

  * how many queries are judgeable (resolvable, >= --min-rows)
  * how many pairs clear --jac-min, and how many anchors get at least one
  * the axis split of those pairs -- the number that actually matters

The axis split is the point. A pair is:

    axis S   different structural_template, same answers
             a rewrite. A fact about the QUERY, learnable from its shape.
    axis C   same template, different constants that share an extension
             a fact about the GRAPH, reachable only through knowledge of what
             the constants denote.

The current DBpedia corpus is 71.6% axis C, which is why a graph encoder cannot
separate itself from a text encoder on it. A corpus worth rebuilding around
should shift that ratio, not merely have more triple patterns.

Usage
-----
    python scripts/check_pair_density.py \
        --queries datasets/dbpedia-structured-manifest.json \
        --result-sets datasets/result_sets_dbpedia_v2.json \
        --compare datasets/dbpedia-triplets-dataset.json
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparqlce.sparql_utils import normalize_query, structural_template


def load_queries(path, field=None):
    text = Path(path).read_text(encoding="utf-8").strip()
    records = json.loads(text) if text.startswith("[") else \
        [json.loads(l) for l in text.splitlines() if l.strip()]
    keys = [field] if field else ["query", "sparql_query", "sparql", "text", "anchor_query"]
    out = []
    for r in records:
        if isinstance(r, str):
            out.append(r)
            continue
        if not isinstance(r, dict):
            continue
        for k in keys:
            if k and r.get(k):
                out.append(str(r[k]))
                break
        for sub in ("positive_queries", "negative_queries"):
            for e in r.get(sub, []) or []:
                q = e.get("query") if isinstance(e, dict) else e
                if q:
                    out.append(str(q))
    return list(dict.fromkeys(normalize_query(q) for q in out if q and q.strip()))


def load_oracle(path):
    """{normalized_query: frozenset(rows)}, minus truncated (prefix) answer sets."""
    if not path.exists():
        raise SystemExit(f"{path} not found.")
    meta = path.with_name(path.stem + "_meta.json")
    bad = ({normalize_query(q) for q in json.load(open(meta)).get("truncated", [])}
           if meta.exists() else set())
    out = {}
    if path.stat().st_size > 512 * 1024 * 1024:
        try:
            import ijson
        except ImportError:
            raise SystemExit("pip install ijson to stream the large oracle.")
        with open(path, "rb") as f:
            for k, v in ijson.kvitems(f, ""):
                nk = normalize_query(k)
                if nk not in bad:
                    out[nk] = frozenset(v)
    else:
        for k, v in json.load(open(path)).items():
            nk = normalize_query(k)
            if nk not in bad:
                out[nk] = frozenset(v)
    return out


def measure(queries, oracle, jac_min, min_rows, label):
    judged = [q for q in queries if len(oracle.get(q, ())) >= min_rows]
    cov = len(judged) / max(len(queries), 1)
    print(f"\n=== {label} ===")
    print(f"  queries {len(queries)} | judgeable at >= {min_rows} rows: "
          f"{len(judged)} ({cov:.1%})")
    if len(judged) < 2:
        print("  too few judgeable queries to form pairs")
        return

    # Inverted index over answer rows: only queries sharing a row can have
    # Jaccard > 0, so the quadratic sweep collapses to the postings.
    postings = collections.defaultdict(list)
    for i, q in enumerate(judged):
        for row in oracle[q]:
            postings[row].append(i)

    tmpl = {q: structural_template(q) for q in judged}
    pos_per_anchor = collections.Counter()
    axis = collections.Counter()
    n_pairs = 0
    hist = collections.Counter()

    for i, q in enumerate(judged):
        a = oracle[q]
        overlap = collections.Counter()
        for row in a:
            for j in postings.get(row, ()):
                if j > i:                      # each unordered pair once
                    overlap[j] += 1
        for j, inter in overlap.items():
            b = oracle[judged[j]]
            jac = inter / (len(a) + len(b) - inter)
            hist[min(int(jac * 10), 9)] += 1
            if jac > jac_min:
                n_pairs += 1
                pos_per_anchor[i] += 1
                pos_per_anchor[j] += 1
                axis["S" if tmpl[judged[j]] != tmpl[q] else "C"] += 1

    n_anch = len(pos_per_anchor)
    print(f"  pairs with Jaccard > {jac_min}: {n_pairs}")
    print(f"  queries with >= 1 such partner: {n_anch} "
          f"({n_anch/max(len(judged),1):.1%} of judgeable)")
    if n_anch:
        vals = sorted(pos_per_anchor.values())
        print(f"  partners per query: median {vals[len(vals)//2]} "
              f"mean {sum(vals)/len(vals):.2f} max {vals[-1]}")
    tot = max(axis["S"] + axis["C"], 1)
    print(f"  AXIS SPLIT: S(rewrite) {axis['S']} ({axis['S']/tot:.1%})  "
          f"C(constants) {axis['C']} ({axis['C']/tot:.1%})")
    print("  Jaccard distribution of overlapping pairs (0.0-0.1 ... 0.9-1.0):")
    print("    " + "  ".join(f"{k/10:.1f}:{hist.get(k,0)}" for k in range(10)))
    return {"anchors": n_anch, "pairs": n_pairs,
            "S": axis["S"], "C": axis["C"], "judgeable": len(judged)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True, help="candidate corpus")
    ap.add_argument("--result-sets", required=True, help="execution oracle")
    ap.add_argument("--compare", default=None,
                    help="second corpus to measure identically, e.g. the current "
                         "datasets/dbpedia-triplets-dataset.json")
    ap.add_argument("--field", default=None)
    ap.add_argument("--jac-min", type=float, default=0.80)
    ap.add_argument("--min-rows", type=int, default=2,
                    help="a one-row answer set makes Jaccard a coin flip between "
                         "1.0 and 0, not a test of set overlap (default 2)")
    args = ap.parse_args()

    orc = load_oracle(Path(args.result_sets) if Path(args.result_sets).is_absolute()
                      else ROOT / args.result_sets)
    print(f"oracle: {len(orc)} queries")

    cand = measure(load_queries(args.queries, args.field), orc,
                   args.jac_min, args.min_rows, Path(args.queries).name)
    base = None
    if args.compare:
        base = measure(load_queries(args.compare), orc,
                       args.jac_min, args.min_rows, Path(args.compare).name)

    if cand and base:
        print("\n=== verdict ===")
        ds = cand["S"] / max(cand["S"] + cand["C"], 1) \
            - base["S"] / max(base["S"] + base["C"], 1)
        print(f"  axis-S share moves {ds:+.1%}")
        print(f"  anchors {base['anchors']} -> {cand['anchors']} "
              f"({cand['anchors']/max(base['anchors'],1):.0%} of the current corpus)")
        if cand["anchors"] < 150:
            print("  WARNING: under ~150 anchors leaves too little to split into "
                  "train/valid/test AND report on; widen the pool (LSQ 2.0) before "
                  "rebuilding around this.")
        if ds <= 0:
            print("  The structural filter did NOT shift the pairs toward rewrites. "
                  "More triple patterns alone does not make a corpus that "
                  "distinguishes a graph encoder from a text encoder.")


if __name__ == "__main__":
    main()
