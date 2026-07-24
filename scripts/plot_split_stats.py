#!/usr/bin/env python3
"""Render the split-profile figure used in datasets/DATASET_STATS.md.

Panels (Wikidata vs DBpedia, test split):
  1. positives-per-anchor histogram (skew + cap at 20)
  2. answer_jaccard composition of positives (=1 vs partial)
  3. answer-set size: singleton fraction (drives the jaccard bimodality)

Run:  python scripts/plot_split_stats.py
Outputs: datasets/figures/split_stats.png
"""
import json
from collections import Counter
from pathlib import Path

import ijson
import matplotlib.pyplot as plt

DS = Path(__file__).resolve().parents[1] / "datasets"
OUT = DS / "figures"
OUT.mkdir(exist_ok=True)

WD, DB = "#4C72B0", "#DD8452"


def load(kg):
    triplets = json.load(open(DS / f"{kg}-triplets-dataset.json"))
    splits = json.load(open(DS / f"splits_{kg}.json"))
    test = [triplets[i] for i in splits["test_idx"]]
    return test


def pos_hist(test):
    b = {"1": 0, "2-4": 0, "5-9": 0, "10-19": 0, "20\n(cap)": 0}
    for x in test:
        c = len(x["positive_queries"])
        if c == 1:
            b["1"] += 1
        elif c <= 4:
            b["2-4"] += 1
        elif c <= 9:
            b["5-9"] += 1
        elif c <= 19:
            b["10-19"] += 1
        else:
            b["20\n(cap)"] += 1
    n = len(test)
    return list(b.keys()), [v / n * 100 for v in b.values()]


def jaccard_split(test):
    eq1 = part = 0
    for x in test:
        for p in x["positive_queries"]:
            if p["answer_jaccard"] == 1.0:
                eq1 += 1
            else:
                part += 1
    tot = eq1 + part
    return eq1 / tot * 100, part / tot * 100


def singleton_frac(kg):
    if kg == "dbpedia":
        d = json.load(open(DS / "result_sets_dbpedia.json"))
        sizes = Counter(len(v) for v in d.values())
    else:
        sizes = Counter()
        with open(DS / "result_sets_wikidata.json", "rb") as fh:
            for _k, arr in ijson.kvitems(fh, ""):
                sizes[len(arr)] += 1
    tot = sum(sizes.values())
    return sizes.get(1, 0) / tot * 100, sum(v for k, v in sizes.items() if k > 1) / tot * 100


def main():
    wd, db = load("wikidata"), load("dbpedia")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # Panel 1: positives per anchor
    labels, wv = pos_hist(wd)
    _, dv = pos_hist(db)
    x = range(len(labels))
    ax = axes[0]
    ax.bar([i - 0.2 for i in x], wv, 0.4, label="Wikidata", color=WD)
    ax.bar([i + 0.2 for i in x], dv, 0.4, label="DBpedia", color=DB)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of anchors")
    ax.set_title("(a) Positives per anchor (test)")
    ax.legend()

    # Panel 2: positive jaccard composition
    ax = axes[1]
    w1, wp = jaccard_split(wd)
    d1, dp = jaccard_split(db)
    ax.bar(["Wikidata", "DBpedia"], [w1, d1], color=[WD, DB], label="jaccard = 1.0 (exact set)")
    ax.bar(["Wikidata", "DBpedia"], [wp, dp], bottom=[w1, d1],
           color=["#A9BEDC", "#EFC3A4"], label="partial (0,1)")
    for i, (a, b) in enumerate([(w1, wp), (d1, dp)]):
        ax.text(i, a / 2, f"{a:.0f}%", ha="center", va="center", color="white", fontweight="bold")
        ax.text(i, a + b / 2, f"{b:.0f}%", ha="center", va="center")
    ax.set_ylabel("% of positives")
    ax.set_title("(b) Positive answer_jaccard composition")
    ax.legend(loc="lower center", fontsize=8)

    # Panel 3: singleton result sets
    ax = axes[2]
    ws, wm = singleton_frac("wikidata")
    dss, dm = singleton_frac("dbpedia")
    ax.bar(["Wikidata", "DBpedia"], [ws, dss], color=[WD, DB], label="singleton (|answers|=1)")
    ax.bar(["Wikidata", "DBpedia"], [wm, dm], bottom=[ws, dss],
           color=["#A9BEDC", "#EFC3A4"], label="multi-element")
    for i, s in enumerate([ws, dss]):
        ax.text(i, s / 2, f"{s:.0f}%", ha="center", va="center", color="white", fontweight="bold")
    ax.set_ylabel("% of result sets")
    ax.set_title("(c) Answer-set size -> jaccard bimodality")
    ax.legend(loc="lower center", fontsize=8)

    fig.suptitle("Wikidata vs DBpedia RQ1 triplet splits — structural profile", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "split_stats.png", dpi=150, bbox_inches="tight")
    print("wrote", OUT / "split_stats.png")


if __name__ == "__main__":
    main()
