#!/usr/bin/env python3
"""RQ1 per-bin evidence figure for the stratified Wikidata split (Option B).

Plots MRR@10 and Hits@10 vs. lexical-similarity bin for the full method roster.
Story: aggregate ranking is dominated by the high-overlap bins (GraphCodeBERT even
edges SparqlCE on the balanced mean), but in the adversarial <0.3 bin — where
retrieval must be result-aware — SparqlCE beats every baseline 3-7x while they all
collapse toward zero. Visual hierarchy:
  * SparqlCE            bold solid blue (the method)
  * E5-Finetuned        dashed blue (same backbone -> isolates the design)
  * GraphCodeBERT       solid orange (aggregate leader that still collapses at <0.3)
  * all other baselines thin grey pack (context)

Numbers are the recorded output of scripts/eval_rq1_stratified.py on
datasets/wikidata_testset_RQ1_stratified.json (SparqlCE ckpt:
checkpoints_b128/wikidata/seed42_full/best_model.pt, e5-base-v2 backbone).

Run:  python scripts/plot_rq1_stratified_results.py
Output: results/figures/rq1_stratified.png  (results/ is gitignored — local)
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BINS = ["<0.3", "0.3-0.5", "0.5-0.8", ">=0.8"]

# full roster: [<0.3, 0.3-0.5, 0.5-0.8, >=0.8]  (ALL in comments)
MRR10 = {
    "TF-IDF":        [0.000, 0.024, 0.292, 0.557],   # 0.218
    "BM25":          [0.000, 0.042, 0.286, 0.531],   # 0.215
    "MiniLM":        [0.004, 0.036, 0.183, 0.289],   # 0.128
    "MPNet":         [0.004, 0.031, 0.152, 0.283],   # 0.117
    "SPBERT":        [0.002, 0.038, 0.154, 0.395],   # 0.147
    "E5 (zs)":       [0.006, 0.040, 0.209, 0.347],   # 0.150
    "CodeBERT":      [0.015, 0.072, 0.294, 0.631],   # 0.253
    "Qwen2.5-Coder": [0.029, 0.077, 0.304, 0.456],   # 0.216
    "GraphCodeBERT": [0.014, 0.083, 0.388, 0.683],   # 0.292
    "E5-FT":         [0.007, 0.050, 0.236, 0.363],   # 0.164
    "SparqlCE":      [0.105, 0.074, 0.315, 0.553],   # 0.262
}
HITS10 = {
    "TF-IDF":        [0.000, 0.071, 0.667, 0.924],   # 0.415
    "BM25":          [0.000, 0.100, 0.638, 0.900],   # 0.410
    "MiniLM":        [0.010, 0.090, 0.410, 0.581],   # 0.273
    "MPNet":         [0.010, 0.095, 0.362, 0.562],   # 0.257
    "SPBERT":        [0.005, 0.105, 0.400, 0.767],   # 0.319
    "E5 (zs)":       [0.019, 0.114, 0.452, 0.700],   # 0.321
    "CodeBERT":      [0.029, 0.167, 0.471, 0.871],   # 0.385
    "Qwen2.5-Coder": [0.052, 0.195, 0.505, 0.686],   # 0.360
    "GraphCodeBERT": [0.029, 0.200, 0.571, 0.886],   # 0.421
    "E5-FT":         [0.019, 0.124, 0.514, 0.705],   # 0.340
    "SparqlCE":      [0.248, 0.186, 0.633, 0.871],   # 0.485
}

GREY = ["TF-IDF", "BM25", "MiniLM", "MPNet", "SPBERT", "E5 (zs)", "CodeBERT", "Qwen2.5-Coder"]
HILITE = {
    "GraphCodeBERT": dict(color="#DD8452", marker="s", ls="-", lw=1.8, ms=6, zorder=4),
    "E5-FT":         dict(color="#8DA5C9", marker="v", ls="--", lw=1.8, ms=6, zorder=4),
    "SparqlCE":      dict(color="#4C72B0", marker="D", ls="-", lw=2.8, ms=9, zorder=5),
}


def panel(ax, data, title, ylabel):
    x = np.arange(len(BINS))
    ax.axvspan(-0.3, 0.3, color="#4C72B0", alpha=0.06)
    # grey baseline pack (one legend proxy)
    for i, name in enumerate(GREY):
        ax.plot(x, data[name], color="#BBBBBB", lw=1.0, alpha=0.7,
                marker="o", ms=3, zorder=2,
                label=f"other baselines (n={len(GREY)})" if i == 0 else None)
    for name, st in HILITE.items():
        ax.plot(x, data[name], label=name, **st)
    ax.text(0, 0.5, "adversarial\n(lexical blind)", color="#4C72B0", fontsize=8,
            ha="center", va="center", transform=ax.get_xaxis_transform())
    ax.set_xticks(x); ax.set_xticklabels(BINS)
    ax.set_xlabel("anchor↔positive lexical similarity (TF-IDF cosine bin)")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3); ax.set_ylim(0, None)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    panel(axes[0], MRR10, "(a) MRR@10 by bin", "MRR@10")
    panel(axes[1], HITS10, "(b) Hits@10 by bin", "Hits@10")
    axes[0].legend(loc="upper left", frameon=True, fontsize=8)
    fig.suptitle("Wikidata RQ1 (stratified, 210/bin): aggregate leaders collapse "
                 "on surface-divergent pairs; only result-aware retrieval holds",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    out = OUT / "rq1_stratified.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
