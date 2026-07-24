# RQ1 — Wikidata stratified split: full results

Test set: `wikidata_testset_RQ1_stratified.json` (840 anchors, 210 per lexical
bin, one result-equivalent positive, 20 hard negatives; corpus 6,212). Single
combined run of `scripts/eval_rq1_stratified.py`:

```bash
python scripts/eval_rq1_stratified.py \
  --st intfloat/e5-base-v2 --st sentence-transformers/all-mpnet-base-v2 \
  --st sentence-transformers/all-MiniLM-L6-v2 --st microsoft/codebert-base \
  --st microsoft/graphcodebert-base --st razent/spbert-mlm-wso-base \
  --st checkpoints/e5_square_wikidata/best_model \
  --code-llm Qwen/Qwen2.5-Coder-0.5B \
  --sparqlce checkpoints_b128/wikidata/seed42_full/best_model.pt
```

SparqlCE ckpt `checkpoints_b128/wikidata/seed42_full/best_model.pt` (e5-base-v2
backbone). Paper tables: `tab-RQ1-Wikidata.tex`. Figure:
`datasets/figures/rq1_stratified.png` (`scripts/plot_rq1_stratified_results.py`).
`Avg` = mean over the three non-trivial bins (`>=0.8` excluded).

## MRR@10 by bin

| Model | <0.3 | 0.3-0.5 | 0.5-0.8 | >=0.8 | Avg(3) |
|---|---:|---:|---:|---:|---:|
| SparqlCE | **0.101** | 0.060 | 0.311 | 0.541 | 0.157 |
| GraphCodeBERT | 0.014 | 0.083 | 0.388 | 0.683 | **0.162** |
| Qwen2.5-Coder-0.5B | 0.029 | 0.077 | 0.304 | 0.456 | 0.137 |
| CodeBERT | 0.015 | 0.072 | 0.294 | 0.631 | 0.127 |
| BM25 | 0.000 | 0.042 | 0.286 | 0.531 | 0.109 |
| TF-IDF | 0.000 | 0.024 | 0.292 | 0.557 | 0.105 |
| E5-Finetuned | 0.007 | 0.050 | 0.236 | 0.363 | 0.098 |
| E5-base-v2 | 0.006 | 0.040 | 0.209 | 0.347 | 0.085 |
| SBERT/MiniLM | 0.004 | 0.036 | 0.183 | 0.289 | 0.074 |
| SPBERT | 0.002 | 0.038 | 0.154 | 0.395 | 0.065 |
| SBERT/MPNet | 0.004 | 0.031 | 0.152 | 0.283 | 0.062 |

## Hits@10 by bin

| Model | <0.3 | 0.3-0.5 | 0.5-0.8 | >=0.8 | Avg(3) |
|---|---:|---:|---:|---:|---:|
| SparqlCE | **0.248** | 0.157 | 0.643 | 0.871 | **0.349** |
| GraphCodeBERT | 0.029 | 0.200 | 0.571 | 0.886 | 0.267 |
| Qwen2.5-Coder-0.5B | 0.052 | 0.195 | 0.505 | 0.686 | 0.251 |
| TF-IDF | 0.000 | 0.071 | 0.667 | 0.924 | 0.246 |
| BM25 | 0.000 | 0.100 | 0.638 | 0.900 | 0.246 |
| CodeBERT | 0.029 | 0.167 | 0.471 | 0.871 | 0.222 |
| E5-Finetuned | 0.019 | 0.124 | 0.514 | 0.705 | 0.219 |
| E5-base-v2 | 0.019 | 0.114 | 0.452 | 0.700 | 0.195 |
| SBERT/MiniLM | 0.010 | 0.090 | 0.410 | 0.581 | 0.170 |
| SPBERT | 0.005 | 0.105 | 0.400 | 0.767 | 0.170 |
| SBERT/MPNet | 0.010 | 0.095 | 0.362 | 0.562 | 0.156 |

## nDCG@10 by bin

| Model | <0.3 | 0.3-0.5 | 0.5-0.8 | >=0.8 | Avg(3) |
|---|---:|---:|---:|---:|---:|
| SparqlCE | **0.135** | 0.083 | 0.390 | 0.621 | **0.203** |
| GraphCodeBERT | 0.017 | 0.111 | 0.432 | 0.732 | 0.187 |
| Qwen2.5-Coder-0.5B | 0.034 | 0.104 | 0.352 | 0.511 | 0.163 |
| CodeBERT | 0.018 | 0.094 | 0.337 | 0.688 | 0.150 |
| BM25 | 0.000 | 0.056 | 0.369 | 0.623 | 0.142 |
| TF-IDF | 0.000 | 0.035 | 0.381 | 0.648 | 0.139 |
| E5-Finetuned | 0.010 | 0.067 | 0.303 | 0.446 | 0.127 |
| E5-base-v2 | 0.009 | 0.057 | 0.268 | 0.433 | 0.111 |
| SBERT/MiniLM | 0.005 | 0.048 | 0.237 | 0.359 | 0.097 |
| SPBERT | 0.003 | 0.053 | 0.212 | 0.485 | 0.089 |
| SBERT/MPNet | 0.005 | 0.046 | 0.203 | 0.351 | 0.085 |

`@3` and `@5` are in `tab-RQ1-Wikidata.tex`.

## Winner summary

- **`<0.3` adversarial bin: SparqlCE wins every metric at every cutoff (@3/@5/@10)**,
  by 3–7× (e.g. MRR@3 0.080 vs next-best Qwen 0.025; lexical methods 0.000).
- **3-bin aggregate:** SparqlCE wins **Hits** (all cutoffs) and **nDCG@5/@10**;
  GraphCodeBERT wins **MRR** (all cutoffs) and **nDCG@3** — its lead coming entirely
  from the high-overlap bins (0.388 at 0.5–0.8, 0.683 at ≥0.8 vs ~0.014 at <0.3).
- **Backbone ablation:** E5-Finetuned (SparqlCE's backbone, fine-tuned) collapses at
  `<0.3` like the zero-shot models (MRR@10 0.007), so the adversarial-bin advantage is
  the result-aware design, not the backbone.

The aggregate MRR inversion (GraphCodeBERT > SparqlCE on average, SparqlCE ≫ on
`<0.3`) is the concrete evidence that a single aggregate misranks models for
result-aware retrieval.
