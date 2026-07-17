#!/usr/bin/env python3
"""Fig. 3 — cosine vs answer-set Jaccard, one panel per model (the direct
alignment figure). Hexbin density (a scatter that stays legible at large
N) with the y=x diagonal; a result-aware model hugs the line, a surface
encoder forms a flat high band.

Real use: call plot_grid(df, model_cols, rho) in the 07_visualize_alignment
notebook, where `df` has a 'jaccard' column + one cosine column per model.
__main__ renders a SYNTHETIC MOCK for design review only.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white", context="paper")
plt.rcParams.update({"font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix"})
PRETTY = {"Qwen3-ft": "Qwen3-Emb (ft)", "Qwen3-zs": "Qwen3-Emb (zs)"}
RED = "#C03A2B"


def plot_grid(df, model_cols, rho=None, ncols=3, gridsize=24,
              out_pdf="alignment_scatter_dbpedia.pdf"):
    """
    df         : DataFrame with 'jaccard' + one cosine column per model.
    model_cols : ordered model names (SparqlCE first reads best).
    rho        : optional {model: spearman_rho} shown in each title.
    """
    from scipy.stats import spearmanr
    if rho is None:
        rho = {m: float(spearmanr(df[m], df["jaccard"]).correlation)
               for m in model_cols}
    panels = [(m, df["jaccard"].values, df[m].values, rho[m]) for m in model_cols]
    _draw(panels, ncols, gridsize, out_pdf)


SUPERVISED = {"SparqlCE", "Qwen3-ft", "Qwen3-Emb (ft)"}
BLUE = "#4C72B0"; GRAY = "#7A7F87"


def _draw(panels, ncols, gridsize, out_pdf):
    n = len(panels); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.35 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    hexes = []
    for k, (m, j, c, rh) in enumerate(panels):
        ax = axes[k // ncols][k % ncols]
        hb = ax.hexbin(j, c, gridsize=gridsize, cmap="rocket_r", bins="log",
                       mincnt=1, extent=(0, 1, 0, 1), linewidths=0.1)
        hexes.append(hb)
    vmax = max(h.get_array().max() for h in hexes)
    for h in hexes:
        h.set_clim(1, vmax)
    for k, (m, j, c, rh) in enumerate(panels):
        ax = axes[k // ncols][k % ncols]
        ax.plot([0, 1], [0, 1], ls=(0, (3, 3)), color="#2b6cb0", lw=1.1, zorder=3)
        name = PRETTY.get(m, m); sup = name in SUPERVISED; ours = (m == "SparqlCE")
        tcol = RED if ours else (BLUE if sup else GRAY)
        ax.set_title(rf"{name}  ($\rho{{=}}{rh:.2f}$)", fontsize=8.5,
                     weight="bold" if sup else "normal", color=tcol)
        # tint the two answer-supervised panels so the cluster reads at a glance
        if sup:
            ax.set_facecolor("#f4f1fb")
            for s in ax.spines.values():
                s.set_color(tcol); s.set_linewidth(1.3)
        else:
            for s in ax.spines.values():
                s.set_color("#bbb")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=7)
        if k // ncols == nrows - 1:
            ax.set_xlabel("Answer-set Jaccard", fontsize=8.5)
        if k % ncols == 0:
            ax.set_ylabel("Cosine sim.", fontsize=8.5)
    # group labels
    fig.text(0.5, 1.005, "answer-supervised (track the diagonal)      "
             "vs.      surface encoders (flat band, no alignment)",
             ha="center", fontsize=8.5, style="italic", color="#444")
    cb = fig.colorbar(hexes[-1], ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    cb.set_label("pair density (log)", fontsize=7.5); cb.ax.tick_params(labelsize=6.5)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print("wrote", out_pdf)


if __name__ == "__main__":   # synthetic mock (DBpedia rho ordering)
    rng = np.random.default_rng(0); N = 3000
    spec = [("SparqlCE", 0.78, 0.14, 0.11, 0.72), ("Qwen3-Emb (ft)", 0.70, 0.20, 0.12, 0.70),
            ("E5", 0.10, 0.84, 0.05, 0.47), ("MPNet", 0.12, 0.82, 0.06, 0.44),
            ("Qwen3-Emb (zs)", 0.13, 0.80, 0.06, 0.41), ("TF-IDF", 0.16, 0.52, 0.11, 0.37)]
    panels = []
    for m, sl, bs, nz, rh in spec:
        j = rng.beta(0.5, 0.5, N); c = np.clip(bs + sl * j + rng.normal(0, nz, N), 0, 1)
        panels.append((m, j, c, rh))
    _draw(panels, 3, 24, "alignment_scatter_dbpedia.pdf")
    print("MOCK rendered (synthetic data)")
