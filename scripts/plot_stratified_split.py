#!/usr/bin/env python3
"""Composition figure for the stratified RQ1 split (Option B).

Panels:
  (a) equal bins: anchors per lexical-similarity bin
  (b) anchor->positive lexical cos per bin (the controlled variable)
  (c) hard negatives vs positives: lexical-cos overlap that makes negatives hard

Run:  python scripts/plot_stratified_split.py
Output: datasets/figures/stratified_split.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DS = Path(__file__).resolve().parents[1] / "datasets"
OUT = DS / "figures"
OUT.mkdir(exist_ok=True)

BINS = ["<0.3", "0.3-0.5", "0.5-0.8", ">=0.8"]
COL = {"<0.3": "#4C72B0", "0.3-0.5": "#5B8FB0", "0.5-0.8": "#C4A35A", ">=0.8": "#DD8452"}


def main():
    d = json.load(open(DS / "wikidata_testset_RQ1_stratified.json"))
    by_bin = {b: [r["lexical_cos"] for r in d if r["lexical_bin"] == b] for b in BINS}
    pos_cos = [r["lexical_cos"] for r in d]
    neg_cos = [n["lexical_cos"] for r in d for n in r["negative_queries"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) counts per bin
    ax = axes[0]
    ax.bar(BINS, [len(by_bin[b]) for b in BINS], color=[COL[b] for b in BINS])
    for i, b in enumerate(BINS):
        ax.text(i, len(by_bin[b]) + 3, str(len(by_bin[b])), ha="center")
    ax.set_ylabel("anchors")
    ax.set_title("(a) Equal bins (210 each, N=840)")
    ax.set_ylim(0, max(len(v) for v in by_bin.values()) * 1.15)

    # (b) positive cos distribution per bin
    ax = axes[1]
    ax.boxplot([by_bin[b] for b in BINS], labels=BINS, patch_artist=True,
               boxprops=dict(facecolor="#A9BEDC"), medianprops=dict(color="black"))
    ax.axhline(0.3, ls="--", c="#DD8452", lw=1)
    ax.text(4.3, 0.31, "cos=0.3", color="#DD8452", ha="right", fontsize=8)
    ax.set_ylabel("anchor↔positive TF-IDF cos")
    ax.set_title("(b) Positive lexical similarity per bin")

    # (c) positives vs hard negatives lexical cos
    ax = axes[2]
    bins = np.linspace(0, 1, 26)
    ax.hist(pos_cos, bins=bins, alpha=0.6, color="#4C72B0", density=True,
            label=f"positives (n={len(pos_cos)})")
    ax.hist(neg_cos, bins=bins, alpha=0.6, color="#DD8452", density=True,
            label=f"hard negatives (n={len(neg_cos)})")
    ax.axvline(np.median(neg_cos), ls="--", c="#DD8452", lw=1)
    ax.set_xlabel("TF-IDF cos to anchor")
    ax.set_ylabel("density")
    ax.set_title("(c) Hard negatives are lexically close")
    ax.legend(fontsize=8)

    fig.suptitle("Wikidata RQ1 stratified split — composition", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "stratified_split.png", dpi=150, bbox_inches="tight")
    print("wrote", OUT / "stratified_split.png")


if __name__ == "__main__":
    main()
