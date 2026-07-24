#!/usr/bin/env python3
"""RQ1 per-bin evidence figure for the stratified Wikidata split (Option B).

Plots MRR@10 and Hits@10 vs. lexical-similarity bin for each method. The story:
lexical (TF-IDF/BM25) and generic text embedding (MPNet) collapse toward zero as
anchor<->positive surface overlap vanishes; the result-aware SparqlCE keeps
non-trivial retrieval on the adversarial <0.3 bin.

Numbers are the recorded output of:
    python scripts/eval_rq1_stratified.py --sparqlce <ckpt> --st <model>
(SparqlCE checkpoint: checkpoints_b128/wikidata/seed42_full/best_model.pt,
e5-base-v2 backbone). Update the dicts below if you re-run.

Run:  python scripts/plot_rq1_stratified_results.py
Output: results/figures/rq1_stratified.png  (results/ is gitignored — local)
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BINS = ["<0.3", "0.3-0.5", "0.5-0.8", ">=0.8"]

MRR10 = {
    "TF-IDF":    [0.000, 0.024, 0.292, 0.557],
    "BM25":      [0.000, 0.042, 0.286, 0.531],
    "MPNet":     [0.004, 0.031, 0.152, 0.283],
    "E5-FT":     [0.007, 0.050, 0.236, 0.363],
    "SparqlCE":  [0.105, 0.074, 0.315, 0.553],
}
HITS10 = {
    "TF-IDF":    [0.000, 0.071, 0.667, 0.924],
    "BM25":      [0.000, 0.100, 0.638, 0.900],
    "MPNet":     [0.010, 0.095, 0.362, 0.562],
    "E5-FT":     [0.019, 0.124, 0.514, 0.705],
    "SparqlCE":  [0.248, 0.186, 0.633, 0.871],
}
STYLE = {
    "TF-IDF":   dict(color="#8C8C8C", marker="o", ls="--"),
    "BM25":     dict(color="#B0B0B0", marker="s", ls="--"),
    "MPNet":    dict(color="#DD8452", marker="^", ls="-."),
    # E5-FT: same backbone as SparqlCE -> same colour family, dashed = weaker method
    "E5-FT":    dict(color="#8DA5C9", marker="v", ls="--", lw=1.8),
    "SparqlCE": dict(color="#4C72B0", marker="D", ls="-", lw=2.5, ms=8),
}


def panel(ax, data, title, ylabel):
    x = np.arange(len(BINS))
    for name, ys in data.items():
        ax.plot(x, ys, label=name, **STYLE[name])
    ax.axvspan(-0.3, 0.3, color="#4C72B0", alpha=0.06)
    ax.text(0, 0.5, "adversarial\n(lexical blind)", color="#4C72B0", fontsize=8,
            ha="center", va="center", transform=ax.get_xaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels(BINS)
    ax.set_xlabel("anchor↔positive lexical similarity (TF-IDF cosine bin)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, None)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    panel(axes[0], MRR10, "(a) MRR@10 by bin", "MRR@10")
    panel(axes[1], HITS10, "(b) Hits@10 by bin", "Hits@10")
    axes[0].legend(loc="upper left", frameon=True)
    fig.suptitle("Wikidata RQ1 (stratified, 210/bin): baselines collapse on "
                 "surface-divergent pairs; result-aware retrieval does not",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    out = OUT / "rq1_stratified.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
