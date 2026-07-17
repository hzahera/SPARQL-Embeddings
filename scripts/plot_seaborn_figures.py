#!/usr/bin/env python3
"""Paper figures in native seaborn gallery style.

Follows the seaborn example gallery conventions:
  sns.set_theme(style="whitegrid"), the "deep" categorical palette,
  pastel/muted two-tone bars (part_whole_bars example), alpha fills,
  and sns.despine(). Default seaborn sans-serif font.

Same numbers as the paper's matplotlib scripts; only the rendering
changes. Outputs PDF (paper) + PNG (preview).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

OUT = os.path.dirname(os.path.abspath(__file__))

sns.set_theme(style="whitegrid", context="paper")
# keep the seaborn palette/theme, but use a serif face to match the paper body
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

# seaborn "deep" palette anchors
DEEP = sns.color_palette("deep")
BLUE, ORANGE, GREEN, RED, PURPLE, BROWN, PINK, GRAY, TAN, CYAN = DEEP
# highlight colour for SparqlCE = deep red; baselines in deep blue / gray
OURS, FT, REST = RED, BLUE, GRAY


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# =====================================================================
# 1. Caching: grouped bars (deep palette) + stability lines
# =====================================================================
THETA = [0.7, 0.8, 0.9]
KGS = {
    "DBpedia": {
        "hit":  {"SparqlCE": [0.843, 0.687, 0.328], "Qwen3-ft": [0.788, 0.601, 0.159]},
        "qual": {"SparqlCE": [0.178, 0.264, 0.354], "Qwen3-ft": [0.171, 0.307, 0.612]},
        "best_of_rest": [0.089, 0.058, 0.083], "ylim": 0.22,
        "panels": ("A", "B"), "collapse": True,
    },
    "Wikidata": {
        "hit":  {"SparqlCE": [0.844, 0.647, 0.545], "Qwen3-ft": [0.640, 0.600, 0.417]},
        "qual": {"SparqlCE": [0.052, 0.077, 0.081], "Qwen3-ft": [0.058, 0.053, 0.119]},
        "best_of_rest": [0.034, 0.041, 0.052], "ylim": 0.065,
        "panels": ("C", "D"), "collapse": False,
    },
}


def caching_fig():
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4))
    pal = {"Best other baseline": REST, "Qwen3-Emb (fine-tuned)": FT, "SparqlCE": OURS}

    for row, (kg, cfg) in enumerate(KGS.items()):
        util = {m: [h * q for h, q in zip(cfg["hit"][m], cfg["qual"][m])] for m in cfg["hit"]}
        rows = []
        for i, t in enumerate(THETA):
            rows += [("Best other baseline", t, cfg["best_of_rest"][i]),
                     ("Qwen3-Emb (fine-tuned)", t, util["Qwen3-ft"][i]),
                     ("SparqlCE", t, util["SparqlCE"][i])]
        du = pd.DataFrame(rows, columns=["Model", "theta", "Utility"])

        axu = axes[row][0]
        sns.barplot(du, x="theta", y="Utility", hue="Model", palette=pal,
                    order=THETA, hue_order=list(pal), ax=axu, alpha=.85,
                    legend=(row == 0))
        axu.set_ylim(0, cfg["ylim"] * (1.16 if row == 0 else 1.0))
        axu.set(xlabel="", ylabel="Utility (hit rate × quality)")
        axu.set_xticklabels([f"θ={t}" for t in THETA])
        axu.set_title(f"{cfg['panels'][0]}. Cache utility — {kg}", weight="bold")
        for cont, m, col in [(axu.containers[1], "Qwen3-ft", FT),
                             (axu.containers[2], "SparqlCE", OURS)]:
            axu.bar_label(cont, labels=[f"{v:.3f}" for v in util[m]], padding=2,
                          fontsize=6.5, color=col)
        if row == 0:
            sns.move_legend(axu, "upper center", ncol=3, frameon=False,
                            fontsize=6.4, title=None, handlelength=1.1,
                            columnspacing=1.0, bbox_to_anchor=(0.5, 1.02))
        sns.despine(ax=axu, left=True)

        axs_ = axes[row][1]
        dl = pd.DataFrame({
            "theta": THETA * 2,
            "Hit rate": cfg["hit"]["SparqlCE"] + cfg["hit"]["Qwen3-ft"],
            "Model": ["SparqlCE"] * 3 + ["Qwen3-Emb (fine-tuned)"] * 3,
        })
        sns.lineplot(dl, x="theta", y="Hit rate", hue="Model", style="Model",
                     palette={"SparqlCE": OURS, "Qwen3-Emb (fine-tuned)": FT},
                     markers={"SparqlCE": "o", "Qwen3-Emb (fine-tuned)": "s"},
                     dashes={"SparqlCE": "", "Qwen3-Emb (fine-tuned)": (4, 2)},
                     linewidth=2.0, markersize=7, ax=axs_, legend=(row == 0))
        axs_.set_ylim(0, 1.0)
        axs_.set_xticks(THETA)
        axs_.set(xlabel="Cache threshold θ", ylabel="Hit rate")
        axs_.set_title(f"{cfg['panels'][1]}. Stability across θ — {kg}", weight="bold")
        for m, col, dy in [("SparqlCE", OURS, 0.05), ("Qwen3-ft", FT, -0.09)]:
            axs_.annotate(f'{cfg["hit"][m][-1]:.3f}', xy=(0.9, cfg["hit"][m][-1]),
                          xytext=(0.855, cfg["hit"][m][-1] + dy), fontsize=7, color=col)
        if cfg["collapse"]:
            axs_.annotate("collapses", xy=(0.9, cfg["hit"]["Qwen3-ft"][-1]),
                          xytext=(0.80, 0.30), fontsize=7, style="italic", color=".35",
                          arrowprops=dict(arrowstyle="->", color=".5", lw=0.8))
        if row == 0:
            sns.move_legend(axs_, "lower left", frameon=False, fontsize=6.6, title=None)
        sns.despine(ax=axs_)

    fig.tight_layout(w_pad=2.4, h_pad=2.4)
    save(fig, "caching_utility_seaborn")


# =====================================================================
# 2. Rank-stability: pastel/muted two-tone bars (part_whole_bars style)
# =====================================================================
MODELS = ["TF-IDF", "MiniLM", "MPNet", "E5", "Qwen3-Emb",
          "Qwen3-Emb\n(fine-tuned)", "SparqlCE"]
RANK = {
    "DBpedia":  {"orig": [0.0688, 0.0614, 0.0835, 0.0752, 0.0774, 0.1312, 0.2802],
                 "pert": [0.0749, 0.0635, 0.0901, 0.0696, 0.0797, 0.1470, 0.2848]},
    "Wikidata": {"orig": [0.0931, 0.1209, 0.1175, 0.1282, 0.1103, 0.2083, 0.2850],
                 "pert": [0.0861, 0.1202, 0.1048, 0.1109, 0.1102, 0.2247, 0.2834]},
}
PASTEL = sns.color_palette("pastel")
MUTED = sns.color_palette("muted")


def rank_fig():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), sharey=True)
    for ax, kg in zip(axes, RANK):
        d = RANK[kg]
        rows = []
        for i, m in enumerate(MODELS):
            rows += [(m, "Original", d["orig"][i]), (m, "Perturbed", d["pert"][i])]
        df = pd.DataFrame(rows, columns=["Model", "Condition", "MRR"])
        # baseline blue two-tone (pastel original / muted perturbed)
        sns.barplot(df, x="Model", y="MRR", hue="Condition", order=MODELS,
                    palette=[PASTEL[0], MUTED[0]], ax=ax,
                    legend=(kg == "DBpedia"))
        # recolour the SparqlCE pair to red two-tone to highlight "ours"
        ax.containers[0].patches[-1].set_facecolor(PASTEL[3])
        ax.containers[1].patches[-1].set_facecolor(MUTED[3])
        ref = d["pert"][-1]
        ax.axhline(ref, color=MUTED[3], ls=(0, (4, 3)), lw=1.0, zorder=0)
        ax.text(3.1, ref + 0.007, "SparqlCE perturbed", fontsize=6.8,
                color=MUTED[3], ha="center")
        ax.bar_label(ax.containers[1],
                     labels=[""] * (len(MODELS) - 1) + [f'{ref:.3f}'],
                     padding=2, fontsize=6.5)
        ax.set_title(kg, weight="bold")
        ax.set(xlabel="")
        ax.set_xticklabels(MODELS, rotation=32, ha="right", fontsize=7.5)
        ax.set_ylim(0, 0.31)
        if kg == "DBpedia":
            sns.move_legend(ax, "upper left", frameon=False, title=None, fontsize=7.5)
        sns.despine(ax=ax, left=True)
    axes[0].set_ylabel("MRR")
    fig.tight_layout(w_pad=1.8)
    save(fig, "rank_stability_seaborn")


# =====================================================================
# 3. Robustness radar (deep palette, alpha fill, whitegrid polar)
# =====================================================================
AXES = ["Var.\nRenaming", "Triple\nReorder", "Redundant\nOPTIONAL", "UNION\nDup.",
        "KG-specific$^\\dagger$", "FILTER\n(true)", "DISTINCT\nToggle"]
RAD = {
    "DBpedia": {
        "SparqlCE (Ours)": [1.000, 1.000, 0.980, 1.000, 1.000, 1.000, 1.000],
        "TF-IDF":          [0.953, 0.997, 0.957, 0.927, 0.989, 0.981, 0.975],
        "E5":              [0.988, 0.999, 0.991, 0.989, 0.998, 0.991, 0.979],
        "Qwen3":           [0.959, 0.995, 0.991, 0.972, 0.989, 0.994, 0.989],
    },
    "Wikidata": {
        "SparqlCE (Ours)": [1.000, 0.995, 0.946, 0.995, 1.000, 1.000, 0.999],
        "TF-IDF":          [0.896, 0.987, 0.946, 0.934, 1.000, 0.932, 0.948],
        "E5":              [0.982, 0.999, 0.996, 0.975, 0.999, 0.991, 0.985],
        "Qwen3":           [0.965, 0.987, 0.994, 0.940, 0.997, 0.996, 0.986],
    },
}
RCOL = {"SparqlCE (Ours)": RED, "E5": BLUE, "Qwen3": GREEN, "TF-IDF": GRAY}
RMK = {"SparqlCE (Ours)": "o", "E5": "^", "Qwen3": "D", "TF-IDF": "s"}
RLS = {"SparqlCE (Ours)": "-", "E5": "-.", "Qwen3": ":", "TF-IDF": "--"}
RMIN, RMAX = 0.85, 1.0


def radar_fig():
    n = len(AXES)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    with sns.axes_style("whitegrid"):
        fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6), subplot_kw=dict(polar=True))
    for ax, kg in zip(axs, RAD):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        for name in ["TF-IDF", "Qwen3", "E5", "SparqlCE (Ours)"]:
            v = np.clip(np.array(RAD[kg][name] + RAD[kg][name][:1]), RMIN, RMAX)
            a = np.concatenate([ang, ang[:1]])
            lw = 2.4 if "Ours" in name else 1.3
            ax.plot(a, v, color=RCOL[name], lw=lw, ls=RLS[name], marker=RMK[name],
                    markersize=4 if "Ours" in name else 3,
                    zorder=5 if "Ours" in name else 3, label=name, clip_on=False)
            if "Ours" in name:
                ax.fill(a, v, color=RCOL[name], alpha=.20, zorder=1)
        ax.set_ylim(RMIN, RMAX)
        ax.set_yticks([0.88, 0.92, 0.96, 1.00])
        ax.set_yticklabels(["0.88", "0.92", "0.96", "1.00"], fontsize=6.5, color=".4")
        ax.set_rlabel_position(12)
        ax.set_xticks(ang)
        ax.set_xticklabels(AXES, fontsize=7.5)
        ax.tick_params(pad=3)
        ax.set_title(kg, fontsize=11, weight="bold", pad=18)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "robustness_radar_seaborn")


if __name__ == "__main__":
    caching_fig()
    rank_fig()
    radar_fig()
    print("done")
