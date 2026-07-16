#!/usr/bin/env python3
"""Generate the caching utility/stability figure (paper Figure: caching_utility.pdf).

Two panels at K=100 on the DBpedia caching stream (source:
notebooks/caching/caching_simulation_dbpedia.ipynb, full grid in
results/caching/caching_dbpedia.json):

  A. Utility (hit rate x quality = expected served Jaccard per query) vs.
     threshold theta, for SparqlCE, the fine-tuned Qwen3 baseline, and the
     best of the remaining (non-fine-tuned) baselines at each theta.
  B. Hit rate vs. theta for SparqlCE and the fine-tuned Qwen3 baseline,
     showing SparqlCE degrading gracefully while the baseline collapses at
     theta=0.9.

Usage:
    python3 scripts/plot_caching_utility.py [output.pdf]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.linewidth": 0.6,
})

THETA = [0.7, 0.8, 0.9]

# K=100 rows from results/caching/caching_dbpedia.json
HIT_RATE = {
    "SparqlCE": [0.843, 0.687, 0.328],
    "Qwen3-ft": [0.788, 0.601, 0.159],
}
QUALITY = {
    "SparqlCE": [0.178, 0.264, 0.354],
    "Qwen3-ft": [0.171, 0.307, 0.612],
}
# Best of the remaining (non-fine-tuned) baselines at each theta: Qwen3-zero-shot
# (0.7), BM25 (0.8), SBERT/MPNet (0.9) -- identity varies by theta, see caption.
BEST_OF_REST = [0.089, 0.058, 0.083]

UTILITY = {
    name: [h * q for h, q in zip(HIT_RATE[name], QUALITY[name])]
    for name in HIT_RATE
}

RED_D = "#C03A2B"      # SparqlCE
BLUE_D = "#4C72B0"     # fine-tuned Qwen3
GRAY_L = "#B9BFC8"     # best of remaining baselines


def main(out_path="caching_utility.pdf"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    # ---- Panel A: utility bar chart -----------------------------------
    x = np.arange(len(THETA))
    w = 0.26
    ax1.bar(x - w, BEST_OF_REST, w, color=GRAY_L, edgecolor="white",
            linewidth=0.7, label="Best other baseline")
    ax1.bar(x, UTILITY["Qwen3-ft"], w, color=BLUE_D, edgecolor="white",
            linewidth=0.7, label="Qwen3-Emb (fine-tuned)")
    ax1.bar(x + w, UTILITY["SparqlCE"], w, color=RED_D, edgecolor="white",
            linewidth=0.7, label="SparqlCE")

    for i, theta in enumerate(THETA):
        ax1.text(x[i] + w, UTILITY["SparqlCE"][i] + 0.004,
                  f'{UTILITY["SparqlCE"][i]:.3f}', ha="center", va="bottom",
                  fontsize=6.5, color=RED_D, fontweight="bold")
        ax1.text(x[i], UTILITY["Qwen3-ft"][i] + 0.004,
                  f'{UTILITY["Qwen3-ft"][i]:.3f}', ha="center", va="bottom",
                  fontsize=6.5, color=BLUE_D)

    ax1.annotate("tie", xy=(x[1], max(UTILITY["SparqlCE"][1], UTILITY["Qwen3-ft"][1]) + 0.018),
                 ha="center", fontsize=6.5, color="#57606A", style="italic")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$\\theta$={t}" for t in THETA], fontsize=7.5)
    ax1.set_ylabel("Utility (hit rate $\\times$ quality)", fontsize=7.5)
    ax1.set_title("A. Cache utility", fontsize=9, fontweight="bold", pad=6)
    ax1.set_ylim(0, 0.22)
    ax1.yaxis.grid(True, color="#E4E7EB", linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="y", labelsize=7, length=2, width=0.5, color="#9AA1AA")
    ax1.tick_params(axis="x", length=0)
    for s in ("top", "right", "left"):
        ax1.spines[s].set_visible(False)
    ax1.spines["bottom"].set_color("#9AA1AA")
    ax1.legend(loc="upper left", frameon=False, fontsize=6.3, handlelength=1.2,
               borderaxespad=0.2)

    # ---- Panel B: hit-rate stability line plot ------------------------
    ax2.plot(THETA, HIT_RATE["SparqlCE"], color=RED_D, marker="o",
             linewidth=1.8, markersize=4, label="SparqlCE")
    ax2.plot(THETA, HIT_RATE["Qwen3-ft"], color=BLUE_D, marker="s",
             linestyle=(0, (4, 2)), linewidth=1.4, markersize=4,
             label="Qwen3-Emb (fine-tuned)")

    ax2.annotate(f'{HIT_RATE["SparqlCE"][-1]:.3f}',
                 xy=(THETA[-1], HIT_RATE["SparqlCE"][-1]),
                 xytext=(THETA[-1] - 0.045, HIT_RATE["SparqlCE"][-1] + 0.04),
                 fontsize=6.5, color=RED_D, fontweight="bold")
    ax2.annotate(f'{HIT_RATE["Qwen3-ft"][-1]:.3f}',
                 xy=(THETA[-1], HIT_RATE["Qwen3-ft"][-1]),
                 xytext=(THETA[-1] - 0.045, HIT_RATE["Qwen3-ft"][-1] - 0.09),
                 fontsize=6.5, color=BLUE_D)
    ax2.annotate("collapses", xy=(THETA[-1], HIT_RATE["Qwen3-ft"][-1]),
                 xytext=(0.82, 0.30), fontsize=6.5, color="#57606A",
                 style="italic", arrowprops=dict(arrowstyle="->", color="#9AA1AA",
                                                  linewidth=0.6))

    ax2.set_xticks(THETA)
    ax2.set_xticklabels([f"{t}" for t in THETA], fontsize=7.5)
    ax2.set_xlabel(r"Cache threshold $\theta$", fontsize=7.5)
    ax2.set_ylabel("Hit rate", fontsize=7.5)
    ax2.set_title("B. Stability across $\\theta$", fontsize=9, fontweight="bold", pad=6)
    ax2.set_ylim(0, 1.0)
    ax2.yaxis.grid(True, color="#E4E7EB", linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="y", labelsize=7, length=2, width=0.5, color="#9AA1AA")
    ax2.tick_params(axis="x", length=2, width=0.5, color="#9AA1AA")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.spines["bottom"].set_color("#9AA1AA")
    ax2.spines["left"].set_color("#9AA1AA")
    ax2.legend(loc="lower left", frameon=False, fontsize=6.3, handlelength=1.6,
               borderaxespad=0.2)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
