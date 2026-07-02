# Runbook: Figure 3 — Cosine Similarity vs. Answer-Set Jaccard

Fills **Figure 3** (`fig:cosine_vs_jaccard`) in the paper, which is currently a
placeholder `\fbox` in `src-latex/Sections/experiments-v2.tex` (around lines
161–173). The figure shows, per model, a hexbin density plot of embedding
cosine similarity (y) vs. answer-set Jaccard overlap (x), with a Spearman ρ
box and a `y = x` reference line. Panels: **SparqlCE, MiniLM, MPNet, E5-base,
Qwen3** (SparqlCE in blue, baselines in grey).

The plotting pipeline already exists — nothing new needs to be written:

| Notebook | KG | Output PDF |
|---|---|---|
| `notebooks/evaluation-RQ1/DBpedia/07_visualize_alignment_dbpedia.ipynb` | DBpedia | `results/cosine_vs_jaccard_rq1.pdf` |
| `notebooks/evaluation-RQ1/Wikidata/07_visualize_alignment_wikidata.ipynb` | Wikidata | `results/cosine_vs_jaccard_rq1_wikidata.pdf` |

Reproducibility: both notebooks now call `random.seed(42)` /
`np.random.seed(42)` before the stratified pair sampling (added 2026-07-03),
so re-runs on any machine produce the same figure.

---

## 0. Machine setup (once)

```bash
git clone https://github.com/hzahera/SPARQL-Embeddings.git
cd SPARQL-Embeddings
git checkout main-refactored

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install seaborn scipy jupyter        # used by the notebooks, not in requirements.txt
```

Notes:
- A GPU is recommended (Qwen3-Embedding-0.6B encoding is slow on CPU) but not
  required — the notebooks fall back to CPU automatically.
- First run downloads ~2.5 GB of HuggingFace models: `all-MiniLM-L6-v2`,
  `all-mpnet-base-v2`, `intfloat/e5-base-v2`, `Qwen/Qwen3-Embedding-0.6B`.

## 1. Restore the SparqlCE checkpoints (REQUIRED — not in git)

The notebooks load trained checkpoints from `sparqlce/models/`, which is not
tracked in the repo. Copy them from the training machine / backup:

```
sparqlce/models/final_model_mrr_0.4487.pt      # DBpedia   (needed for the main figure)
sparqlce/models/wikidata_model_mrr_0.3238.pt   # Wikidata  (only for the optional Wikidata run)
```

Without these files the notebooks fail in the "Initialize Models" cell.
Everything else (test sets, cached result sets) is already in `datasets/`.

## 2. Run the DBpedia notebook

```bash
jupyter notebook notebooks/evaluation-RQ1/DBpedia/07_visualize_alignment_dbpedia.ipynb
```

Run all cells top-to-bottom. What it does:

1. Loads the 337-anchor `datasets/dbpedia_testset_RQ1.json` and the cached
   result sets `datasets/result_sets_dbpedia.json` (1,677 queries). Queries
   missing from the cache are executed live against
   `https://dbpedia.data.dice-research.org/sparql` — expect this to take a
   while if the cache is incomplete; results are re-cached.
2. Stratified-samples ~30 candidate pairs per anchor across five Jaccard bins
   (seeded, see above) → roughly 10k (anchor, candidate, jaccard) pairs.
3. Encodes the corpus with all five models and builds the 1×5 hexbin panel
   with per-panel Spearman ρ.
4. Saves `results/cosine_vs_jaccard_rq1.pdf` (300 dpi, print-ready).

**Record the five Spearman ρ values printed on the panels** — they go into
the prose (step 5).

## 3. (Optional) Run the Wikidata notebook

Same procedure with
`notebooks/evaluation-RQ1/Wikidata/07_visualize_alignment_wikidata.ipynb`;
endpoint is `https://query.wikidata.org/sparql`, output is
`results/cosine_vs_jaccard_rq1_wikidata.pdf`.

⚠️ **Caution:** an earlier save cell in the Wikidata notebook also writes to
`results/cosine_vs_jaccard_rq1.pdf` (same path as the DBpedia figure). Either
run the Wikidata notebook **before** the DBpedia one, or back up the DBpedia
PDF first.

## 4. Wire the PDF into the paper

```bash
cp results/cosine_vs_jaccard_rq1.pdf  <path-to>/src-latex/images/cosine_vs_jaccard.pdf
```

In `src-latex/Sections/experiments-v2.tex` (~lines 161–173), replace the
placeholder figure with:

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{images/cosine_vs_jaccard.pdf}
  \caption{Embedding geometry: cosine similarity vs.\ answer-set
  Jaccard overlap on DBpedia. \textsc{SparqlCE}'s cosine similarity tracks
  Jaccard across $[0,1]$; baselines do not.}
  \label{fig:cosine_vs_jaccard}
\end{figure*}
```

Changes vs. the placeholder:
- `figure` → **`figure*`** and `\columnwidth` → **`\textwidth`**: the plot is
  five panels at 19.5×3.6 in — at single-column width each panel would be
  ~0.6 in and unreadable. Figure 4 (`robustness_radar`) already uses
  `figure*`, so this is consistent. (Alternative if single-column is required:
  regenerate with only 3 panels — SparqlCE, E5, Qwen3 — by trimming
  `plot_config` in the plotting cell.)
- Added "on DBpedia" to the caption (or show both KGs by stacking the two
  PDFs into a 2-row figure).
- Delete the `% TODO: replace with the actual...` comment and the `\fbox{...}`.

## 5. Sync the prose

- In the paragraph above Figure 3 ("The embedding geometry explains why…"),
  quote the actual Spearman ρ values, e.g. *"ρ = 0.xx for SparqlCE vs.
  ρ ≤ 0.xx for the best baseline"* — the WSDM assessment flags unquantified
  geometry claims.
- The panel set omits TF-IDF (no dense embedding in this pipeline); make sure
  the caption/prose wording of "baselines" matches the five shown models.

## Troubleshooting

- **`FileNotFoundError: sparqlce/models/...`** → step 1 was skipped; the
  checkpoints are not in git.
- **Endpoint timeouts / HTTP 429** during live query execution → re-run the
  cell; results are cached incrementally, so progress is not lost. The
  DBpedia cache in `datasets/result_sets_dbpedia.json` should already cover
  the full corpus.
- **CUDA OOM while encoding with Qwen3** → set `DEVICE = 'cpu'` in the
  "Initialize Models" cell for the Qwen3 encoder only, or reduce the encode
  batch size in `HFEncoder.encode`.
