#!/usr/bin/env python3
"""Seaborn re-style of the cosine-vs-Jaccard alignment figure (paper Fig. 5
and the Wikidata appendix twin).

These panels need the per-pair (cosine, Jaccard) values that only exist
after the six models encode the test corpus, so they cannot be rebuilt
from committed aggregates — run this INSIDE
`notebooks/evaluation-RQ1/*/07_visualize_alignment_*.ipynb`, right after
the cell that builds `df`. `df` must have a 'jaccard' column plus one
cosine-similarity column per model (the same frame the current
`ax.hexbin(df['jaccard'], df[name])` cell consumes).

Drop-in: replace the "Print-ready hexbin panel" cell with a call to
`plot_alignment(df, model_cols, rho, out_pdf)`.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set_theme(context="paper", style="white", font="serif")
mpl.rcParams.update({
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})


def plot_alignment(df, model_cols, rho=None, out_pdf="cosine_vs_jaccard.pdf",
                   gridsize=32, ncols=None):
    """
    df         : DataFrame with a 'jaccard' column + one column per model.
    model_cols : ordered list of model column names (SparqlCE first reads best).
    rho        : optional {model: spearman_rho} to print in each panel title.
    """
    n = len(model_cols)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.35 * ncols, 2.5 * nrows),
                             squeeze=False, sharex=True, sharey=True)

    # unify the density scale across panels for honest visual comparison
    hexes = []
    for k, name in enumerate(model_cols):
        ax = axes[k // ncols][k % ncols]
        hb = ax.hexbin(df["jaccard"].values, df[name].values, gridsize=gridsize,
                       cmap="rocket_r", mincnt=1, bins="log", linewidths=0.15,
                       extent=(0, 1, 0, 1))
        hexes.append(hb)
    vmax = max(h.get_array().max() for h in hexes)
    for h in hexes:
        h.set_clim(1, vmax)

    for k, name in enumerate(model_cols):
        ax = axes[k // ncols][k % ncols]
        ax.plot([0, 1], [0, 1], ls="--", color="#C03A2B", lw=1.0, zorder=3)
        title = name if rho is None else rf"{name}  ($\rho$={rho[name]:.2f})"
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=7.5)
        if k % ncols == 0:
            ax.set_ylabel("Cosine similarity", fontsize=8)
        if k // ncols == nrows - 1:
            ax.set_xlabel("Answer-set Jaccard", fontsize=8)
        sns.despine(ax=ax)

    # single shared colorbar
    cb = fig.colorbar(hexes[-1], ax=axes.ravel().tolist(), fraction=0.025,
                      pad=0.012)
    cb.set_label("pair density (log)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    fig.savefig(out_pdf, bbox_inches="tight")
    print("saved", out_pdf)
    return fig


# --- alternative: single-panel jointplot with KDE marginals (SparqlCE only) ---
def plot_alignment_joint(df, model_col="SparqlCE", out_pdf="alignment_joint.pdf"):
    g = sns.jointplot(data=df, x="jaccard", y=model_col, kind="hex",
                      color="#C03A2B", height=4.2,
                      marginal_kws=dict(bins=40, fill=True),
                      joint_kws=dict(gridsize=32, bins="log", cmap="rocket_r"))
    g.ax_joint.plot([0, 1], [0, 1], ls="--", color="#1F2328", lw=1.0)
    g.set_axis_labels("Answer-set Jaccard", f"Cosine similarity ({model_col})",
                      fontsize=9)
    g.savefig(out_pdf, bbox_inches="tight")
    print("saved", out_pdf)
    return g
