#!/usr/bin/env python3
"""Per-lexical-bin RQ1 report from exported per-anchor scores, across seeds.

Answers "does the margin hold where surface form carries least signal?" -- the
regime a result-aware encoder is supposed to own. Anchors are binned by the mean
TF-IDF cosine between the anchor and its positives, so `<0.3` is the adversarial
stratum: lexically dissimilar queries that are nonetheless result-equivalent.

Two things this reports that a single-number stratified table cannot:

  * the SEED SPREAD inside each bin. A bin holding 20-odd anchors moves a lot
    between training runs, and a per-bin margin smaller than that spread is not a
    finding. The `<0.3` bin of the DBpedia split holds 22 of 219 anchors.
  * a seed-level paired test per bin, when exactly two --scores globs are given.

Binning is done on the anchor's positives as they appear in the split. Use
--bins-on original to bin on only the pre-relabelling positives (entries without
a `relabelled` flag), which keeps bin membership comparable to results published
before the split was relabelled. On DBpedia the two agree to within one anchor.

Usage
-----
    python scripts/report_by_lexical_bin.py \
        --split datasets/dbpedia-testsplit-RQ1.json \
        --scores "results/significance/dbpedia-testsplit-RQ1-relabelled_sparqlce_seed*.json" \
        --scores "results/significance/dbpedia-testsplit-RQ1-relabelled_e5ft_seed*.json"

    # just one bin, all metrics
    python scripts/report_by_lexical_bin.py --split ... --scores ... --only "<0.3"
"""
import argparse
import glob as globmod
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_norm

BIN_EDGES = [("<0.3", 0.0, 0.3), ("0.3-0.5", 0.3, 0.5),
             ("0.5-0.8", 0.5, 0.8), (">=0.8", 0.8, 1.01)]
METRICS = ["mrr@3", "mrr@5", "mrr@10", "ndcg@3", "ndcg@5", "ndcg@10"]


def norm(s):
    return " ".join(str(s).split())


def anchor_cosines(split, use_relabelled=True):
    """Mean anchor->positive TF-IDF cosine per anchor, in split order."""
    texts, idx = [], {}

    def add(q):
        n = norm(q)
        if n not in idx:
            idx[n] = len(texts)
            texts.append(n)

    for r in split:
        add(r["anchor_query"])
        for p in r["positive_queries"]:
            add(p["query"])
    X = sk_norm(TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
                .fit_transform(texts), norm="l2")
    out = []
    for r in split:
        ps = [p for p in r["positive_queries"]
              if use_relabelled or not p.get("relabelled")]
        ai = idx[norm(r["anchor_query"])]
        cs = [float(X[ai].multiply(X[idx[norm(p['query'])]]).sum()) for p in ps]
        out.append(float(np.mean(cs)) if cs else 0.0)
    return np.array(out)


def load_model(pattern):
    """Per-seed per-anchor score matrices for one model: {metric: (n_seeds, n_anchors)}."""
    files = sorted(globmod.glob(pattern))
    if not files:
        raise SystemExit(f"no score files matched {pattern}")
    anchors, label, seeds, per = None, None, [], {}
    for f in files:
        d = json.load(open(f))
        a = [norm(x) for x in d["anchors"]]
        if anchors is None:
            anchors, label = a, d.get("model_id") or Path(f).stem
        elif a != anchors:
            raise SystemExit(f"{f}: anchor list differs from the other seeds")
        seeds.append(d.get("seed"))
        for m in METRICS:
            if m in d.get("per_anchor", {}):
                per.setdefault(m, []).append(np.array(d["per_anchor"][m], dtype=float))
    mats = {m: np.vstack(v) for m, v in per.items()}
    # Hits@k is derivable: per-anchor MRR@k > 0 iff a positive appeared in top-k.
    for k in (3, 5, 10):
        if f"mrr@{k}" in mats:
            mats[f"hits@{k}"] = (mats[f"mrr@{k}"] > 0).astype(float)
    return label, seeds, anchors, mats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--scores", action="append", default=[], metavar="GLOB",
                    help="one model per glob (repeatable; seeds averaged)")
    ap.add_argument("--bins-on", default="all", choices=["all", "original"],
                    help="bin on all positives (default) or only pre-relabelling ones")
    ap.add_argument("--only", default=None, help="report a single bin, e.g. '<0.3'")
    ap.add_argument("--metrics", default="mrr@10,ndcg@10,hits@10",
                    help="comma-separated (default mrr@10,ndcg@10,hits@10; "
                         "use 'all' for every available metric)")
    args = ap.parse_args()

    split = json.load(open(args.split))
    cos = anchor_cosines(split, use_relabelled=(args.bins_on == "all"))
    split_anchors = [norm(r["anchor_query"]) for r in split]

    models = [load_model(p) for p in args.scores]
    if not models:
        raise SystemExit("--scores is required")

    # Align the bin vector to each model's anchor order.
    pos = {a: i for i, a in enumerate(split_anchors)}
    aligned = []
    for label, seeds, anchors, mats in models:
        if anchors != split_anchors:
            if not set(anchors) <= set(pos):
                raise SystemExit(f"{label}: anchors are not a subset of the split")
            c = np.array([cos[pos[a]] for a in anchors])
        else:
            c = cos
        aligned.append((label, seeds, c, mats))

    wanted = (list(models[0][3].keys()) if args.metrics == "all"
              else [m.strip() for m in args.metrics.split(",")])
    bins = [b for b in BIN_EDGES if args.only is None or b[0] == args.only]
    if not bins:
        raise SystemExit(f"--only {args.only!r} is not one of "
                         f"{[b[0] for b in BIN_EDGES]}")

    print(f"split: {Path(args.split).name}  anchors={len(split)}  "
          f"bins on: {args.bins_on} positives")
    for label, seeds, _, _ in aligned:
        print(f"  model: {label}  seeds={seeds}")

    for name, lo, hi in bins:
        mask0 = (aligned[0][2] >= lo) & (aligned[0][2] < hi)
        n = int(mask0.sum())
        print(f"\n{'=' * 72}\nBIN {name}   n={n} anchors "
              f"({n / len(split):.1%} of the split)\n{'=' * 72}")
        if n == 0:
            print("  (empty)")
            continue
        if n < 30:
            print(f"  NOTE: {n} anchors. Per-seed spread below is the relevant scale;")
            print(f"        a margin smaller than it is not a result at this size.")
        print(f"  {'metric':<9} " + " ".join(f"{lab[:22]:>24}"
              for lab, _, _, _ in aligned) + f"{'diff':>10} {'seed p':>8}")
        for m in wanted:
            cells, seed_means = [], []
            for label, seeds, c, mats in aligned:
                if m not in mats:
                    cells.append(f"{'n/a':>24}")
                    seed_means.append(None)
                    continue
                mask = (c >= lo) & (c < hi)
                per_seed = mats[m][:, mask].mean(axis=1)      # one value per seed
                seed_means.append(per_seed)
                cells.append(f"{per_seed.mean():9.4f}+-{per_seed.std(ddof=1):<6.4f}"
                             if len(per_seed) > 1 else f"{per_seed.mean():16.4f}")
            tail = ""
            if len(seed_means) == 2 and all(s is not None for s in seed_means):
                a, b = seed_means
                d = a.mean() - b.mean()
                if len(a) > 2 and len(a) == len(b):
                    t = ttest_rel(a, b)
                    p = t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2
                    tail = f"{d:+10.4f} {p:8.3f}"
                else:
                    tail = f"{d:+10.4f} {'n<3':>8}"
            print(f"  {m:<9} " + " ".join(cells) + tail)

        if len(aligned) == 2:
            a_lab, a_seeds, a_c, a_mats = aligned[0]
            b_lab, b_seeds, b_c, b_mats = aligned[1]
            if "mrr@10" in a_mats and "mrr@10" in b_mats:
                ma = (a_c >= lo) & (a_c < hi)
                mb = (b_c >= lo) & (b_c < hi)
                pa = a_mats["mrr@10"][:, ma].mean(axis=1)
                pb = b_mats["mrr@10"][:, mb].mean(axis=1)
                print(f"  per-seed MRR@10  first: {[round(v, 4) for v in pa]}")
                print(f"                  second: {[round(v, 4) for v in pb]}")
                print(f"  first model wins {int((pa > pb).sum())}/{len(pa)} seeds "
                      f"in this bin")


if __name__ == "__main__":
    main()
