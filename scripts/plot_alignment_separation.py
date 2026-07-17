#!/usr/bin/env python3
"""Fig. 3 — embedding alignment as an overlap-separation dumbbell.

For each model, plot mean cosine similarity on low-overlap pairs (J<0.2,
open marker) vs high-overlap pairs (J>0.8, filled marker); the bar length
is how much the embedding geometry separates result-different from
result-equivalent queries. Result-aware models stretch across the axis;
surface encoders sit as short stubs high on the right.

Real use: call plot_dumbbell(df, model_cols) inside the
07_visualize_alignment notebook, where `df` has a 'jaccard' column plus
one cosine column per model (the same frame the hexbin used). The
__main__ renders a SYNTHETIC MOCK for design review only.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix"})
D = sns.color_palette("deep")
OURS, FT, MUT = D[3], D[0], "#8A8F98"
PRETTY = {"Qwen3-ft": "Qwen3-Emb (ft)", "Qwen3-zs": "Qwen3-Emb (zs)"}


def _colr(m):
    return OURS if m == "SparqlCE" else (FT if m in ("Qwen3-ft", "Qwen3-Emb (ft)") else MUT)


def plot_dumbbell(df, model_cols, low=0.2, high=0.8,
                  out_pdf="alignment_separation_dbpedia.pdf", figsize=(3.5, 3.1)):
    """
    df         : DataFrame with 'jaccard' + one cosine column per model.
    model_cols : model column names to plot.
    Computes, per model, mean cosine on pairs with jaccard<low and jaccard>high.
    """
    lo_mask = df["jaccard"] < low
    hi_mask = df["jaccard"] > high
    rows = []
    for m in model_cols:
        lo = float(df.loc[lo_mask, m].mean())
        hi = float(df.loc[hi_mask, m].mean())
        rows.append((PRETTY.get(m, m), lo, hi))
    rows.sort(key=lambda r: r[2] - r[1])          # biggest separation on top
    _draw(rows, out_pdf, figsize)


def _draw(rows, out_pdf, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    for i, (m, lo, hi) in enumerate(rows):
        c = _colr(m); ours = m == "SparqlCE"
        ax.plot([lo, hi], [i, i], color=c, lw=3.2 if ours else 2.0, zorder=2,
                solid_capstyle="round")
        ax.scatter(lo, i, s=52, facecolor="white", edgecolor=c, linewidth=1.6, zorder=3)
        ax.scatter(hi, i, s=62, color=c, zorder=3)
        ax.text(hi + 0.015, i, f"+{hi - lo:.2f}", va="center", fontsize=7,
                color=c, fontweight="bold" if ours else "normal")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([m for m, *_ in rows], fontsize=8.5)
    for t, (m, *_) in zip(ax.get_yticklabels(), rows):
        if m == "SparqlCE":
            t.set_color(OURS); t.set_fontweight("bold")
        elif m == "Qwen3-Emb (ft)":
            t.set_color(FT)
    lows = [r[1] for r in rows] + [r[2] for r in rows]
    ax.set_xlim(max(0, min(lows) - 0.1), 1.0)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("Mean cosine similarity", fontsize=9)
    ax.set_title("Cosine separation of low- vs high-overlap pairs",
                 fontsize=9.5, weight="bold", pad=8)
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                  markeredgecolor="#444", markersize=8, label="low overlap ($J{<}0.2$)"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="#444",
                  markersize=8, label="high overlap ($J{>}0.8$)")]
    ax.legend(handles=leg, frameon=False, fontsize=7.3, loc="lower right",
              handletextpad=0.3, labelspacing=0.25)
    sns.despine(ax=ax, left=True)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=210, bbox_inches="tight")
    print("wrote", out_pdf)


if __name__ == "__main__":   # synthetic mock for design review only
    _draw([("E5", 0.86, 0.90), ("Qwen3-Emb (zs)", 0.80, 0.86),
           ("MPNet", 0.82, 0.88), ("TF-IDF", 0.52, 0.62),
           ("Qwen3-Emb (ft)", 0.35, 0.83), ("SparqlCE", 0.30, 0.85)][::-1],
          "alignment_separation_dbpedia.pdf", (3.5, 3.1))
    print("MOCK rendered (synthetic data)")
