#!/usr/bin/env python3
"""Statistical profiling of the Wikidata vs DBpedia triplet splits.

Motivation: the Wikidata positives/anchor and answer_jaccard distributions look
"abnormal" (jaccard seems to be either 0 or 1; positives-per-anchor is skewed).
This script quantifies that per split (train/valid/test) so we can describe how
the datasets actually look, and shows DBpedia for contrast.

Run:  python scripts/analyze_split_stats.py
"""
import json
import statistics as st
from pathlib import Path

DS = Path(__file__).resolve().parents[1] / "datasets"


def load(kg):
    triplets = json.load(open(DS / f"{kg}-triplets-dataset.json"))
    splits = json.load(open(DS / f"splits_{kg}.json"))
    return triplets, splits


def q(vals):
    vals = sorted(vals)
    n = len(vals)

    def pct(p):
        if n == 0:
            return float("nan")
        i = min(n - 1, int(round(p / 100 * (n - 1))))
        return vals[i]

    return dict(
        n=n,
        min=vals[0],
        p25=pct(25),
        median=st.median(vals),
        mean=sum(vals) / n,
        p75=pct(75),
        p95=pct(95),
        max=vals[-1],
    )


def jaccard_profile(jvals):
    """Bimodality check: fractions exactly 0, exactly 1, and strictly between."""
    n = len(jvals)
    if n == 0:
        return {}
    exact0 = sum(1 for v in jvals if v == 0.0)
    exact1 = sum(1 for v in jvals if v == 1.0)
    between = n - exact0 - exact1
    return dict(
        n=n,
        frac_eq0=exact0 / n,
        frac_eq1=exact1 / n,
        frac_between=between / n,
        mean=sum(jvals) / n,
    )


def analyze(kg):
    triplets, splits = load(kg)
    print("=" * 70)
    print(f"{kg.upper()}  —  {len(triplets)} total anchors")
    print("=" * 70)

    for split in ["train_idx", "valid_idx", "test_idx"]:
        idx = splits[split]
        items = [triplets[i] for i in idx]
        npos = [len(x["positive_queries"]) for x in items]
        nneg = [len(x["negative_queries"]) for x in items]

        pos_j, neg_j = [], []
        for x in items:
            pos_j += [p["answer_jaccard"] for p in x["positive_queries"]]
            neg_j += [p["answer_jaccard"] for p in x["negative_queries"]]

        name = split.replace("_idx", "")
        print(f"\n[{name}]  {len(items)} anchors")

        s = q(npos)
        print(
            f"  positives/anchor : min {s['min']}  p25 {s['p25']}  "
            f"median {s['median']}  mean {s['mean']:.2f}  p75 {s['p75']}  "
            f"p95 {s['p95']}  max {s['max']}"
        )
        # how many anchors have exactly 1 positive
        one = sum(1 for c in npos if c == 1)
        print(f"                     anchors with exactly 1 positive: "
              f"{one}/{len(npos)} ({one/len(npos):.1%})")

        s = q(nneg)
        print(
            f"  negatives/anchor : min {s['min']}  median {s['median']}  "
            f"mean {s['mean']:.2f}  max {s['max']}"
        )

        # histogram of positives/anchor + truncation at the cap (20)
        buckets = {"1": 0, "2-4": 0, "5-9": 0, "10-19": 0, "20 (cap)": 0}
        for c in npos:
            if c == 1:
                buckets["1"] += 1
            elif c <= 4:
                buckets["2-4"] += 1
            elif c <= 9:
                buckets["5-9"] += 1
            elif c <= 19:
                buckets["10-19"] += 1
            else:
                buckets["20 (cap)"] += 1
        hist = "  ".join(
            f"{k}:{v} ({v/len(npos):.0%})" for k, v in buckets.items()
        )
        print(f"                     hist -> {hist}")

        pj = jaccard_profile(pos_j)
        nj = jaccard_profile(neg_j)
        print(
            f"  positive jaccard : n {pj['n']}  =0 {pj['frac_eq0']:.1%}  "
            f"in(0,1) {pj['frac_between']:.1%}  =1 {pj['frac_eq1']:.1%}  "
            f"mean {pj['mean']:.3f}"
        )
        print(
            f"  negative jaccard : n {nj['n']}  =0 {nj['frac_eq0']:.1%}  "
            f"in(0,1) {nj['frac_between']:.1%}  =1 {nj['frac_eq1']:.1%}  "
            f"mean {nj['mean']:.3f}"
        )


if __name__ == "__main__":
    for kg in ["wikidata", "dbpedia"]:
        analyze(kg)
        print()
