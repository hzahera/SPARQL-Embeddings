#!/usr/bin/env python3
"""Generate the caching utility/stability figure (paper Figure: caching_utility.pdf).

Two rows of two panels at K=100 — DBpedia (top) and Wikidata (bottom).
Sources: notebooks/caching/caching_simulation_dbpedia.ipynb and its Wikidata
twin; full grids in results/caching/caching_dbpedia.json and
results/caching/caching_wikidata.json.

  A/C. Utility (hit rate x quality = expected served Jaccard per query) vs.
       threshold theta, for SparqlCE, the fine-tuned Qwen3 baseline, and the
       best of the remaining (non-fine-tuned) baselines at each theta.
       Utility y-scales differ per graph: the Wikidata stream offers far less
       reusable overlap (median best-available Jaccard 0.0 vs 0.25).
  B/D. Hit rate vs. theta for SparqlCE and the fine-tuned Qwen3 baseline.

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

# K=100 rows from results/caching/caching_{dbpedia,wikidata}.json
KGS = {
    "DBpedia": {
        "hit_rate": {
            "SparqlCE": [0.843, 0.687, 0.328],
            "Qwen3-ft": [0.788, 0.601, 0.159],
        },
        "quality": {
            "SparqlCE": [0.178, 0.264, 0.354],
            "Qwen3-ft": [0.171, 0.307, 0.612],
        },
        # Best remaining baseline utility at each theta: Qwen3 zero-shot (0.7),
        # BM25 (0.8), SBERT/MPNet (0.9) -- identity varies by theta, see caption.
        "best_of_rest": [0.089, 0.058, 0.083],
        "utility_ylim": 0.22,
        "panels": ("A", "B"),
        "annotate_tie_at": 1,        # theta=0.8 (index)
        "annotate_collapse": True,   # fine-tuned baseline collapses at 0.9
    },
    "Wikidata": {
        "hit_rate": {
            "SparqlCE": [0.844, 0.647, 0.545],
            "Qwen3-ft": [0.640, 0.600, 0.417],
        },
        "quality": {
            "SparqlCE": [0.052, 0.077, 0.081],
            "Qwen3-ft": [0.058, 0.053, 0.119],
        },
        # Best remaining baseline utility at each theta: TF-IDF (0.7),
        # Qwen3 zero-shot (0.8), SBERT/MPNet (0.9).
        "best_of_rest": [0.034, 0.041, 0.052],
        "utility_ylim": 0.065,
        "panels": ("C", "D"),
        "annotate_tie_at": None,
        "annotate_collapse": False,
    },
}

RED_D = "#C03A2B"      # SparqlCE
BLUE_D = "#4C72B0"     # fine-tuned Qwen3
GRAY_L = "#B9BFC8"     # best of remaining baselines


def draw_utility_panel(ax, kg_name, cfg, show_legend):
    hit, qual = cfg["hit_rate"], cfg["quality"]
    utility = {m: [h * q for h, q in zip(hit[m], qual[m])] for m in hit}

    x = np.arange(len(THETA))
    w = 0.26
    ax.bar(x - w, cfg["best_of_rest"], w, color=GRAY_L, edgecolor="white",
           linewidth=0.7, label="Best other baseline")
    ax.bar(x, utility["Qwen3-ft"], w, color=BLUE_D, edgecolor="white",
           linewidth=0.7, label="Qwen3-Emb (fine-tuned)")
    ax.bar(x + w, utility["SparqlCE"], w, color=RED_D, edgecolor="white",
           linewidth=0.7, label="SparqlCE")

    pad = cfg["utility_ylim"] * 0.02
    for i in range(len(THETA)):
        ax.text(x[i] + w, utility["SparqlCE"][i] + pad,
                f'{utility["SparqlCE"][i]:.3f}', ha="center", va="bottom",
                fontsize=6.5, color=RED_D, fontweight="bold")
        ax.text(x[i], utility["Qwen3-ft"][i] + pad,
                f'{utility["Qwen3-ft"][i]:.3f}', ha="center", va="bottom",
                fontsize=6.5, color=BLUE_D)

    if cfg["annotate_tie_at"] is not None:
        i = cfg["annotate_tie_at"]
        top = max(utility["SparqlCE"][i], utility["Qwen3-ft"][i])
        ax.annotate("tie", xy=(x[i], top + cfg["utility_ylim"] * 0.082),
                    ha="center", fontsize=6.5, color="#57606A", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\theta$={t}" for t in THETA], fontsize=7.5)
    ax.set_ylabel("Utility (hit rate $\\times$ quality)", fontsize=7.5)
    ax.set_title(f"{cfg['panels'][0]}. Cache utility — {kg_name}",
                 fontsize=9, fontweight="bold", pad=6)
    ax.set_ylim(0, cfg["utility_ylim"])
    ax.yaxis.grid(True, color="#E4E7EB", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=7, length=2, width=0.5, color="#9AA1AA")
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#9AA1AA")
    if show_legend:
        ax.legend(loc="upper left", frameon=False, fontsize=6.3,
                  handlelength=1.2, borderaxespad=0.2)


def draw_stability_panel(ax, kg_name, cfg, show_legend):
    hit = cfg["hit_rate"]
    ax.plot(THETA, hit["SparqlCE"], color=RED_D, marker="o",
            linewidth=1.8, markersize=4, label="SparqlCE")
    ax.plot(THETA, hit["Qwen3-ft"], color=BLUE_D, marker="s",
            linestyle=(0, (4, 2)), linewidth=1.4, markersize=4,
            label="Qwen3-Emb (fine-tuned)")

    ax.annotate(f'{hit["SparqlCE"][-1]:.3f}',
                xy=(THETA[-1], hit["SparqlCE"][-1]),
                xytext=(THETA[-1] - 0.045, hit["SparqlCE"][-1] + 0.04),
                fontsize=6.5, color=RED_D, fontweight="bold")
    ax.annotate(f'{hit["Qwen3-ft"][-1]:.3f}',
                xy=(THETA[-1], hit["Qwen3-ft"][-1]),
                xytext=(THETA[-1] - 0.045, hit["Qwen3-ft"][-1] - 0.09),
                fontsize=6.5, color=BLUE_D)
    if cfg["annotate_collapse"]:
        ax.annotate("collapses", xy=(THETA[-1], hit["Qwen3-ft"][-1]),
                    xytext=(0.82, 0.30), fontsize=6.5, color="#57606A",
                    style="italic",
                    arrowprops=dict(arrowstyle="->", color="#9AA1AA",
                                    linewidth=0.6))

    ax.set_xticks(THETA)
    ax.set_xticklabels([f"{t}" for t in THETA], fontsize=7.5)
    ax.set_xlabel(r"Cache threshold $\theta$", fontsize=7.5)
    ax.set_ylabel("Hit rate", fontsize=7.5)
    ax.set_title(f"{cfg['panels'][1]}. Stability across $\\theta$ — {kg_name}",
                 fontsize=9, fontweight="bold", pad=6)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, color="#E4E7EB", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=7, length=2, width=0.5, color="#9AA1AA")
    ax.tick_params(axis="x", length=2, width=0.5, color="#9AA1AA")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#9AA1AA")
    ax.spines["left"].set_color("#9AA1AA")
    if show_legend:
        ax.legend(loc="lower left", frameon=False, fontsize=6.3,
                  handlelength=1.6, borderaxespad=0.2)


def main(out_path="caching_utility.pdf"):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))

    for row, (kg_name, cfg) in enumerate(KGS.items()):
        draw_utility_panel(axes[row][0], kg_name, cfg, show_legend=(row == 0))
        draw_stability_panel(axes[row][1], kg_name, cfg, show_legend=(row == 0))

    fig.tight_layout(w_pad=2.5, h_pad=2.2)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
