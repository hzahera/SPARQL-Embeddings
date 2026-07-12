#!/usr/bin/env python3
"""Generate the RQ2 rank-stability bar chart (paper Figure: rank_stability_bars.pdf).

Plots MRR on original vs. perturbed queries (full-corpus ranking, anchor
excluded) for all baselines and SparqlCE on DBpedia and Wikidata. The values
are the ones reported in the paper's rank-stability table (tab:rq2_mrr),
produced by notebooks/evaluation-RQ2/{DBpedia,Wikidata}/04_eval_sparql_RQ2_*.ipynb
and 09_eval_qwen3_finetuned_RQ2_*.ipynb.

Usage:
    python3 scripts/plot_rank_stability.py [output.pdf]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.linewidth": 0.6,
})

MODELS = ["TF-IDF", "MiniLM", "MPNet", "E5", "Qwen3-Emb", "Qwen3-Emb\n(fine-tuned)", "SparqlCE"]
DATA = {
    "DBpedia":  {"orig": [0.0688, 0.0614, 0.0835, 0.0752, 0.0774, 0.1312, 0.2802],
                 "pert": [0.0749, 0.0635, 0.0901, 0.0696, 0.0797, 0.1470, 0.2848]},
    "Wikidata": {"orig": [0.0931, 0.1209, 0.1175, 0.1282, 0.1103, 0.2083, 0.2850],
                 "pert": [0.0861, 0.1202, 0.1048, 0.1109, 0.1102, 0.2247, 0.2834]},
}

GRAY_L, GRAY_D = "#B9BFC8", "#6E7681"   # baselines: original (hatched) / perturbed
RED_L,  RED_D  = "#E39A8E", "#C03A2B"   # SparqlCE:  original (hatched) / perturbed


def main(out_path="rank_stability_bars.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), sharey=True)
    x = np.arange(len(MODELS))
    w = 0.38

    for ax, kg in zip(axes, DATA):
        d = DATA[kg]
        n = len(MODELS)
        ax.bar(x - w/2, d["orig"], w, color=[GRAY_L]*(n-1) + [RED_L], hatch="///",
               edgecolor="white", linewidth=0.7)
        ax.bar(x + w/2, d["pert"], w, color=[GRAY_D]*(n-1) + [RED_D],
               edgecolor="white", linewidth=0.7)

        # headline reference: SparqlCE perturbed vs every baseline original
        ref = d["pert"][-1]
        ax.axhline(ref, color=RED_D, linestyle=(0, (4, 3)), linewidth=0.9, zorder=0)
        ax.annotate("SparqlCE perturbed", xy=(-0.45, ref), xytext=(-0.45, ref - 0.020),
                    fontsize=6.5, color=RED_D, ha="left")

        # selective direct labels: SparqlCE pair + fine-tuned Qwen3 perturbed
        ax.text(x[-1] - w/2 - 0.12, d["orig"][-1] + 0.005, f'{d["orig"][-1]:.3f}',
                ha="center", va="bottom", fontsize=6.2, color="#1F2328")
        ax.text(x[-1] + w/2 + 0.12, d["pert"][-1] + 0.005, f'{d["pert"][-1]:.3f}',
                ha="center", va="bottom", fontsize=6.2, color="#1F2328")
        ax.text(x[-2] + w/2, d["pert"][-2] + 0.005, f'{d["pert"][-2]:.3f}',
                ha="center", va="bottom", fontsize=6.2, color="#57606A")

        ax.set_title(kg, fontsize=9, fontweight="bold", pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=32, ha="right", fontsize=7)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=7, length=2, width=0.5, color="#9AA1AA")
        ax.set_ylim(0, 0.29)
        ax.yaxis.grid(True, color="#E4E7EB", linewidth=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#9AA1AA")

    axes[0].set_ylabel("MRR", fontsize=8)
    leg = [
        mpatches.Patch(facecolor=GRAY_L, hatch="///", edgecolor="white", label="Original query"),
        mpatches.Patch(facecolor=GRAY_D, edgecolor="white", label="Perturbed query"),
    ]
    axes[0].legend(handles=leg, loc="upper left", frameon=False, fontsize=7,
                   handlelength=1.4, handleheight=1.1, borderaxespad=0.2)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:2])