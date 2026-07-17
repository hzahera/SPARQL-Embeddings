#!/usr/bin/env python3
"""Drop-in generators for the three model-dependent additions a WSDM
reviewer asked for: (1) a qualitative retrieval table, (2) a t-SNE/UMAP
embedding visualization, (3) an error analysis by SPARQL operator.

These need your TRAINED model's actual outputs (embeddings, rankings,
query strings), which is why they live here rather than being hard-coded:
run them inside the evaluation notebooks where those objects already
exist. Each function is annotated with exactly what to pass. Styling
matches the paper (serif + seaborn `deep` palette).
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
DEEP = sns.color_palette("deep")


# =====================================================================
# 1. Qualitative retrieval table (anchor -> top-k by MiniLM vs SparqlCE)
#    ChatGPT called this "probably the biggest missing experiment".
# =====================================================================
def qualitative_retrieval_table(anchor, rows_minilm, rows_sparqlce,
                                out_tex="qualitative.tex", k=5):
    """
    anchor       : str, the anchor query (short SPARQL, escape for LaTeX).
    rows_minilm  : list of (query_str, answer_jaccard) top-k by MiniLM.
    rows_sparqlce: list of (query_str, answer_jaccard) top-k by SparqlCE.
    Produces a LaTeX table you can \\input. Pick an anchor where SparqlCE
    wins (high-Jaccard neighbors) and MiniLM retrieves lexical look-alikes
    with low answer overlap -- that contrast is the whole point.
    """
    def esc(s):
        return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_")
                 .replace("{", r"\{").replace("}", r"\}").replace("#", r"\#")
                 .replace("&", r"\&").replace("%", r"\%"))

    def block(rows):
        out = []
        for q, j in rows[:k]:
            out.append(rf"\texttt{{\small {esc(q)}}} & {j:.2f} \\")
        return "\n".join(out)

    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Qualitative retrieval for one anchor. MiniLM returns lexical
look-alikes with low answer overlap; \textsc{{SparqlCE}} returns
result-equivalent queries. Right column: answer-set Jaccard with the
anchor.}}
\label{{tab:qualitative}}
\small
\begin{{tabular}}{{@{{}}p{{0.80\columnwidth}} c@{{}}}}
\toprule
\multicolumn{{2}}{{@{{}}l}}{{\textbf{{Anchor:}} \texttt{{\small {esc(anchor)}}}}} \\
\midrule
\multicolumn{{2}}{{@{{}}l}}{{\textit{{Top-{k} retrieved by MiniLM}}}} \\
\midrule
{block(rows_minilm)}
\midrule
\multicolumn{{2}}{{@{{}}l}}{{\textit{{Top-{k} retrieved by \textsc{{SparqlCE}} (ours)}}}} \\
\midrule
{block(rows_sparqlce)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    with open(out_tex, "w") as f:
        f.write(tex)
    print("wrote", out_tex)


# =====================================================================
# 2. Embedding visualization (t-SNE or UMAP, colored by Jaccard-to-anchor)
# =====================================================================
def embedding_scatter(Z, jaccard_to_anchor, method="tsne",
                      out_pdf="embedding_map.pdf", anchor_idx=None):
    """
    Z                : (N, d) array of SparqlCE embeddings for one anchor's
                       candidate pool (unit-normalized is fine).
    jaccard_to_anchor: (N,) answer-set Jaccard of each candidate with the
                       anchor -> used as the color (the property we claim
                       the geometry encodes).
    method           : "tsne" (no extra deps) or "umap" (needs umap-learn).
    """
    if method == "umap":
        import umap
        XY = umap.UMAP(n_neighbors=15, min_dist=0.1,
                       metric="cosine", random_state=0).fit_transform(Z)
    else:
        from sklearn.manifold import TSNE
        XY = TSNE(n_components=2, metric="cosine", init="pca",
                  perplexity=30, random_state=0).fit_transform(Z)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    sc = ax.scatter(XY[:, 0], XY[:, 1], c=jaccard_to_anchor, cmap="rocket_r",
                    s=14, linewidths=0, alpha=0.85)
    if anchor_idx is not None:
        ax.scatter(XY[anchor_idx, 0], XY[anchor_idx, 1], marker="*",
                   s=180, color=DEEP[0], edgecolor="white", linewidth=0.8,
                   zorder=5, label="anchor")
        ax.legend(frameon=False, fontsize=8, loc="best")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("answer-set Jaccard to anchor", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"SparqlCE embedding space ({method.upper()})",
                 fontsize=9, weight="bold")
    sns.despine(ax=ax, left=True, bottom=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    print("wrote", out_pdf)


# =====================================================================
# 3. Error analysis: retrieval quality by SPARQL operator
# =====================================================================
OPERATORS = ["OPTIONAL", "FILTER", "UNION", "DISTINCT",
             "ORDER BY", "LIMIT", "GROUP BY", "aggregate"]


def error_analysis_by_operator(anchor_queries, per_anchor_mrr,
                               out_pdf="error_by_operator.pdf"):
    """
    anchor_queries : list[str] of anchor query strings.
    per_anchor_mrr : (N,) MRR@10 of SparqlCE for each anchor.
    Buckets anchors by which operators their query contains and plots mean
    MRR per bucket -- surfaces where SparqlCE is weakest (e.g. OPTIONAL,
    aggregation), i.e. the failure modes a reviewer wants to see.
    """
    import pandas as pd
    rows = []
    up = [q.upper() for q in anchor_queries]
    for op in OPERATORS:
        mask = np.array([op.upper() in q for q in up])
        if mask.sum() == 0:
            continue
        rows.append((op, float(np.mean(np.asarray(per_anchor_mrr)[mask])),
                     int(mask.sum())))
    df = pd.DataFrame(rows, columns=["Operator", "MRR@10", "n"]).sort_values("MRR@10")

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    sns.barplot(df, y="Operator", x="MRR@10", color=DEEP[0], alpha=.85, ax=ax)
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(r["MRR@10"] + 0.003, i, f'n={r["n"]}', va="center", fontsize=6.5,
                color=".35")
    ax.set_title("Retrieval quality by operator present in anchor",
                 fontsize=9, weight="bold")
    ax.set_ylabel("")
    sns.despine(ax=ax)
    fig.savefig(out_pdf, bbox_inches="tight")
    print("wrote", out_pdf)
    return df
