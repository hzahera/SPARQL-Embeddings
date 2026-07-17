#!/usr/bin/env python3
"""Fig. 3 — embedding alignment as a low- vs high-overlap box plot.

For each model, show the DISTRIBUTION of cosine similarity on low-overlap
pairs (J<0.2) and high-overlap pairs (J>0.8). Result-aware models produce
two disjoint distributions (similarity discriminates answer overlap at the
instance level); surface encoders bunch both regimes into one high band.

Real use: call plot_box(df, model_cols) in the 07_visualize_alignment
notebook, where `df` has a 'jaccard' column plus one cosine column per
model. The __main__ renders a SYNTHETIC MOCK for design review only.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix"})
PRETTY = {"Qwen3-ft": "Qwen3-Emb (ft)", "Qwen3-zs": "Qwen3-Emb (zs)"}
PAL = {"low ($J{<}0.2$)": "#B9C6DB", "high ($J{>}0.8$)": sns.color_palette("deep")[0]}


def plot_box(df, model_cols, low=0.2, high=0.8,
             out_pdf="alignment_separation_dbpedia.pdf", figsize=(3.6, 3.4)):
    lo = df["jaccard"] < low
    hi = df["jaccard"] > high
    rows = []
    for m in model_cols:
        name = PRETTY.get(m, m)
        for v in df.loc[lo, m].values:
            rows.append((name, "low ($J{<}0.2$)", v))
        for v in df.loc[hi, m].values:
            rows.append((name, "high ($J{>}0.8$)", v))
    long = pd.DataFrame(rows, columns=["Model", "Overlap", "cos"])
    # order by separation of medians (biggest on top)
    sep = long.pivot_table(index="Model", columns="Overlap", values="cos",
                           aggfunc="median")
    order = (sep["high ($J{>}0.8$)"] - sep["low ($J{<}0.2$)"]).sort_values(
        ascending=False).index.tolist()
    _draw(long, order, out_pdf, figsize)


def _draw(long, order, out_pdf, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(long, y="Model", x="cos", hue="Overlap", order=order,
                hue_order=list(PAL), palette=PAL, fliersize=0, linewidth=0.8,
                width=0.7, ax=ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Cosine similarity", fontsize=9)
    ax.set_ylabel("")
    ax.set_title("Cosine distribution: low- vs high-overlap pairs",
                 fontsize=9.5, weight="bold", pad=8)
    ax.legend(frameon=False, fontsize=7.3, loc="lower left", title=None)
    for t in ax.get_yticklabels():
        if t.get_text() == "SparqlCE":
            t.set_fontweight("bold")
    sns.despine(ax=ax, left=True)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=210, bbox_inches="tight")
    print("wrote", out_pdf)


if __name__ == "__main__":   # synthetic mock for design review only
    rng = np.random.default_rng(0)
    spec = [("SparqlCE", 0.30, 0.12, 0.85, 0.08),
            ("Qwen3-Emb (ft)", 0.35, 0.13, 0.83, 0.09),
            ("E5", 0.86, 0.04, 0.90, 0.03), ("MPNet", 0.82, 0.05, 0.88, 0.04),
            ("Qwen3-Emb (zs)", 0.80, 0.05, 0.86, 0.04),
            ("TF-IDF", 0.52, 0.10, 0.62, 0.09)]
    rows = []
    for m, lm, ls, hm, hs in spec:
        for v in np.clip(rng.normal(lm, ls, 400), 0, 1):
            rows.append((m, "low ($J{<}0.2$)", v))
        for v in np.clip(rng.normal(hm, hs, 400), 0, 1):
            rows.append((m, "high ($J{>}0.8$)", v))
    long = pd.DataFrame(rows, columns=["Model", "Overlap", "cos"])
    order = ["SparqlCE", "Qwen3-Emb (ft)", "TF-IDF", "MPNet",
             "Qwen3-Emb (zs)", "E5"]
    _draw(long, order, "alignment_separation_dbpedia.pdf", (3.6, 3.4))
    print("MOCK rendered (synthetic data)")
