#!/usr/bin/env python3
"""Redesign of Fig. 3: one panel, one line per model.

Instead of per-model hexbin density panels, bin the answer-set Jaccard
axis and plot each model's MEAN cosine similarity per bin (a calibration
/ alignment curve). A result-aware model tracks y=x; a surface-driven one
stays flat near 1.0. All models share one axis with a legend, and it fits
a single column.

Real use: call plot_alignment_curve(df, model_cols, rho) inside the
07_visualize_alignment notebook, where `df` already has a 'jaccard'
column plus one cosine column per model. The __main__ below renders a
SYNTHETIC MOCK so the design can be reviewed without the trained model.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
DEEP = sns.color_palette("deep")
# style per model: (color, linewidth, linestyle, marker, is_ours)
STYLE = {
    "SparqlCE":      (DEEP[3], 2.4, "-",  "o", True),   # red, bold
    "Qwen3-ft":      (DEEP[0], 1.6, "-",  "s", False),  # blue
    "E5":            (DEEP[2], 1.3, "--", "^", False),  # green
    "MPNet":         (DEEP[4], 1.3, "-.", "D", False),  # purple
    "Qwen3-zs":      (DEEP[1], 1.3, ":",  "v", False),  # orange
    "TF-IDF":        (DEEP[7], 1.3, "--", "P", False),  # gray
}
PRETTY = {"Qwen3-ft": "Qwen3-Emb (ft)", "Qwen3-zs": "Qwen3-Emb (zs)"}


def plot_alignment_curve(df, model_cols, rho=None, nbins=10, band=True,
                         out_pdf="alignment_curve.pdf", figsize=(3.5, 3.0)):
    """
    df         : DataFrame with 'jaccard' + one cosine column per model.
    model_cols : ordered model names (keys of STYLE; SparqlCE last = on top).
    rho        : optional {model: spearman_rho}, shown in the legend.
    """
    edges = np.linspace(0, 1, nbins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    df = df.copy()
    df["_bin"] = np.clip(np.digitize(df["jaccard"], edges) - 1, 0, nbins - 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], ls=(0, (3, 3)), color=".45", lw=1.0, zorder=1,
            label="perfect ($y{=}x$)")
    for m in model_cols:
        col, lw, ls, mk, ours = STYLE.get(m, (DEEP[7], 1.2, "-", "o", False))
        g = df.groupby("_bin")[m]
        mean = g.mean().reindex(range(nbins))
        if band:
            sd = g.std().reindex(range(nbins))
            ax.fill_between(centers, mean - sd, mean + sd, color=col,
                            alpha=0.10 if ours else 0.05, zorder=2)
        lbl = PRETTY.get(m, m)
        if rho and m in rho:
            lbl += rf"  ($\rho{{=}}{rho[m]:.2f}$)"
        ax.plot(centers, mean, color=col, lw=lw, ls=ls, marker=mk,
                markersize=5 if ours else 3.5, zorder=5 if ours else 3,
                label=lbl)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Answer-set Jaccard overlap", fontsize=9)
    ax.set_ylabel("Mean cosine similarity", fontsize=9)
    ax.legend(frameon=False, fontsize=6.6, loc="lower right", handlelength=1.6,
              labelspacing=0.3)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print("wrote", out_pdf)


# ---------------------------------------------------------------- MOCK
def _synth(n=4000, slope=0.7, base=0.2, noise=0.12, floor=0.0, ceil=1.0):
    j = np.random.beta(0.5, 0.5, n)          # near-binary-ish, like Wikidata/DBpedia
    c = base + slope * j + np.random.normal(0, noise, n)
    return j, np.clip(c, floor, ceil)


if __name__ == "__main__":
    np.random.seed(0)
    N = 4000
    jac = np.random.beta(0.5, 0.5, N)
    def cos(slope, base, noise):
        return np.clip(base + slope * jac + np.random.normal(0, noise, N), 0, 1)
    df = pd.DataFrame({
        "jaccard":  jac,
        "SparqlCE": cos(0.72, 0.18, 0.11),   # tracks y=x
        "Qwen3-ft": cos(0.66, 0.24, 0.12),   # tracks, slightly less
        "E5":       cos(0.10, 0.86, 0.05),   # flat, compressed high
        "MPNet":    cos(0.12, 0.82, 0.06),
        "Qwen3-zs": cos(0.14, 0.80, 0.06),
        "TF-IDF":   cos(0.16, 0.55, 0.10),   # lower flat
    })
    rho = {"SparqlCE": 0.72, "Qwen3-ft": 0.70, "E5": 0.47, "MPNet": 0.44,
           "Qwen3-zs": 0.41, "TF-IDF": 0.37}
    order = ["TF-IDF", "Qwen3-zs", "MPNet", "E5", "Qwen3-ft", "SparqlCE"]
    plot_alignment_curve(df, order, rho=rho, out_pdf="alignment_curve_mock.pdf")
    print("MOCK rendered (synthetic data — design preview only)")
