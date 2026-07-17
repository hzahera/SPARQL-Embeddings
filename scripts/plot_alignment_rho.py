#!/usr/bin/env python3
"""Fig. 3 — alignment as Spearman correlation (rho) per model.

For each model, rho between its cosine similarity and the answer-set
Jaccard overlap: how strongly the embedding geometry tracks answer
overlap. One horizontal bar per model, sorted.

Real use: call plot_rho_bars(df, model_cols) in the 07_visualize_alignment
notebook, where `df` has a 'jaccard' column + one cosine column per model.
The __main__ renders a SYNTHETIC MOCK for design review only.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix"})
PRETTY = {"Qwen3-ft": "Qwen3-Emb (ft)", "Qwen3-zs": "Qwen3-Emb (zs)"}
SUP, SUR, OURS = "#4C72B0", "#9AA1AA", "#C03A2B"
SUPERVISED = {"SparqlCE", "Qwen3-ft", "Qwen3-Emb (ft)"}


def _color(m):
    if m == "SparqlCE":
        return OURS
    return SUP if m in SUPERVISED else SUR


def plot_rho_bars(df, model_cols, out_pdf="alignment_separation_dbpedia.pdf",
                  figsize=(3.6, 3.0)):
    from scipy.stats import spearmanr
    rows = [(PRETTY.get(m, m), float(spearmanr(df[m], df["jaccard"]).correlation))
            for m in model_cols]
    _draw(rows, out_pdf, figsize)


def _draw(rows, out_pdf, figsize):
    rows = sorted(rows, key=lambda r: r[1])          # ascending -> best on top
    names = [r[0] for r in rows]
    rhos = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(names, rhos, color=[_color(m) for m in names], alpha=0.95)
    for i, v in enumerate(rhos):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, max(rhos) * 1.18)
    ax.set_xlabel(r"Alignment with answer overlap (Spearman $\rho$)", fontsize=8.5)
    ax.set_title("Embedding alignment per model", fontsize=9.5, weight="bold", pad=8)
    for t in ax.get_yticklabels():
        t.set_fontsize(8.5)
        if t.get_text() == "SparqlCE":
            t.set_fontweight("bold")
    sns.despine(ax=ax, left=True)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=210, bbox_inches="tight")
    print("wrote", out_pdf)


if __name__ == "__main__":   # synthetic mock (DBpedia rho ordering)
    _draw([("SparqlCE", 0.72), ("Qwen3-Emb (ft)", 0.70), ("E5", 0.47),
           ("MPNet", 0.44), ("Qwen3-Emb (zs)", 0.41), ("TF-IDF", 0.37)],
          "alignment_separation_dbpedia.pdf", (3.6, 3.0))
    print("MOCK rendered (synthetic data)")
