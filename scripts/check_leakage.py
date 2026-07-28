#!/usr/bin/env python3
"""Audit an RQ1 split for memorized anchor-positive pairs, and measure what
that memorization is worth to each model.

Why this exists
---------------
`splits_{kg}.json` partitions ANCHOR GROUPS, not query text. The same SPARQL
query recurs across groups, so a held-out group can restate a pair that was
optimized over during training. On the first `wikidata_testset_RQ1_clean.json`
build, 74.8% of eval positive pairs were also train positive pairs.

Both models train on the same pairs, so this is not an unfairness between them
-- but it does not affect them equally, and the split-by-leakage breakdown is
the number that decides whether a retrieval claim is a retrieval claim or a
memorization claim:

    SparqlCE  MRR@10  leaked 0.795   non-leaked 0.577
    E5-FT     MRR@10  leaked 0.475   non-leaked 0.521
    gap               leaked +0.321  non-leaked +0.057

Usage
-----
    # just measure a split
    python scripts/check_leakage.py --kg wikidata \
        --split datasets/wikidata_testset_RQ1_clean.json

    # and break exported per-anchor scores down by leaked/non-leaked
    python scripts/check_leakage.py --kg wikidata \
        --split datasets/wikidata_testset_RQ1_clean.json \
        --scores "results/significance/wikidata_testset_RQ1_clean_sparqlce_seed*.json" \
        --scores "results/significance/wikidata_testset_RQ1_clean_e5ft_seed*.json"

Each --scores glob is one model (averaged over the seeds it matches).

Exit codes (so a caller can gate on leakage without mistaking a crash for a
clean bill of health): 0 = clean, 2 = leakage found, 1 = the audit itself failed
(missing dataset, unreadable split, mismatched anchors).

Rebuild a leak-free split with:
    python scripts/build_clean_rq1.py --kg <kg> --from-split <split> \
        --min-result-size 0 --leak-filter pair
"""
import argparse
import glob as globmod
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

TRIPLETS = {"dbpedia": "dbpedia-triplets-dataset.json",
            "wikidata": "wikidata-triplets-dataset.json"}
SPLITS = {"dbpedia": "splits_dbpedia.json", "wikidata": "splits_wikidata.json"}


def norm(s):
    return " ".join(str(s).split())


def _q(entry):
    if isinstance(entry, dict):
        return str(entry.get("query") or entry.get("sparql") or "")
    return str(entry or "")


def trained_pairs(kg, which="train+valid"):
    """Pairs a model was optimized over, reconstructed exactly as the trainers
    build their datasets (item needs >=1 positive AND >=1 negative to survive)."""
    trip = json.load(open(ROOT / "datasets" / TRIPLETS[kg]))
    spl = json.load(open(ROOT / "datasets" / SPLITS[kg]))
    groups = []
    for item in trip:
        a = item.get("anchor_query")
        if not a:
            continue
        pos = [_q(p) for p in item.get("positive_queries", []) if _q(p)]
        neg = [_q(n) for n in item.get("negative_queries", []) if _q(n)]
        if not pos or not neg:
            continue
        groups.append((norm(a), [norm(p) for p in pos]))
    keys = ["train_idx"] + (["valid_idx"] if which == "train+valid" else [])
    pairs = set()
    for k in keys:
        for i in spl.get(k, []):
            if i < len(groups):
                a, pos = groups[i]
                pairs.update(frozenset((a, p)) for p in pos)
    return pairs, keys


def load_scores(pattern, metric):
    """Mean per-anchor score vector over every seed matching the glob."""
    files = sorted(globmod.glob(pattern))
    if not files:
        raise SystemExit(f"no score files matched {pattern}")
    vecs, anchors, label = [], None, None
    for f in files:
        d = json.load(open(f))
        if metric not in d.get("per_anchor", {}):
            raise SystemExit(f"{f} has no per_anchor['{metric}']")
        vecs.append(np.array(d["per_anchor"][metric], dtype=float))
        a = [norm(x) for x in d["anchors"]]
        if anchors is None:
            anchors, label = a, d.get("model_id") or Path(f).stem
        elif a != anchors:
            raise SystemExit(f"{f}: anchor list differs from the other seeds")
    return np.mean(vecs, axis=0), anchors, label, len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg", required=True, choices=sorted(TRIPLETS))
    ap.add_argument("--split", required=True)
    ap.add_argument("--leak-splits", default="train+valid",
                    choices=["train", "train+valid"])
    ap.add_argument("--scores", action="append", default=[], metavar="GLOB",
                    help="glob of eval_rq1_export.py outputs for ONE model "
                         "(repeatable; seeds are averaged)")
    ap.add_argument("--metric", default="mrr@10")
    args = ap.parse_args()

    split = json.load(open(args.split))
    pairs, keys = trained_pairs(args.kg, args.leak_splits)
    print(f"split: {Path(args.split).name}  anchors={len(split)}")
    print(f"seen pairs in {'+'.join(keys)}: {len(pairs)}")

    leaked_anchor, n_pairs, n_leaked = [], 0, 0
    for it in split:
        a = norm(it["anchor_query"])
        hit = False
        for p in it["positive_queries"]:
            n_pairs += 1
            if frozenset((a, norm(_q(p)))) in pairs:
                n_leaked += 1
                hit = True
        leaked_anchor.append(hit)
    leaked_anchor = np.array(leaked_anchor)

    print(f"\nLEAKAGE")
    print(f"  positive pairs trained on : {n_leaked}/{n_pairs} "
          f"({n_leaked / max(n_pairs, 1):.1%})")
    print(f"  anchors with >=1 such pair: {int(leaked_anchor.sum())}/{len(split)} "
          f"({leaked_anchor.mean():.1%})")
    if n_leaked == 0:
        print("  CLEAN -- no eval label was trained on.")

    if args.scores:
        if leaked_anchor.all() or not leaked_anchor.any():
            print(f"\n(all anchors fall in one group -- no {args.metric} "
                  f"breakdown to compute)")
        else:
            split_anchors = [norm(it["anchor_query"]) for it in split]
            print(f"\n{args.metric.upper()} BY LEAKAGE  "
                  f"(leaked n={int(leaked_anchor.sum())}, "
                  f"non-leaked n={int((~leaked_anchor).sum())})")
            print(f"  {'model':<44} {'leaked':>8} {'non-leaked':>11} {'all':>8}")
            rows = OrderedDict()
            for pattern in args.scores:
                v, anchors, label, n = load_scores(pattern, args.metric)
                if anchors != split_anchors:
                    # Reorder the mask to the export's anchor order rather than
                    # assuming the split order was preserved.
                    idx = {a: i for i, a in enumerate(split_anchors)}
                    if not set(anchors) <= set(idx):
                        raise SystemExit(
                            f"{pattern}: anchors are not a subset of {args.split} "
                            f"-- these scores were exported against a different split")
                    mask = np.array([leaked_anchor[idx[a]] for a in anchors])
                else:
                    mask = leaked_anchor
                rows[f"{label} ({n} seeds)"] = (v[mask].mean(), v[~mask].mean(), v.mean())
                print(f"  {label[:43]:<44} {v[mask].mean():>8.4f} "
                      f"{v[~mask].mean():>11.4f} {v.mean():>8.4f}")
            if len(rows) == 2:
                (a_l, a_n, a_a), (b_l, b_n, b_a) = rows.values()
                print(f"  {'gap (first - second)':<44} {a_l - b_l:>+8.4f} "
                      f"{a_n - b_n:>+11.4f} {a_a - b_a:>+8.4f}")
                print(f"\n  The non-leaked column is the retrieval result; the "
                      f"leaked column is\n  a memorization result. Report the "
                      f"former, or report both.")

    return 2 if n_leaked else 0


if __name__ == "__main__":
    sys.exit(main())
