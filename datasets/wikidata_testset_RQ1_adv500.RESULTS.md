# RQ1 — Wikidata adversarial-500: results

Test set: `wikidata_testset_RQ1_adv500.json` — 500 anchors, each with ≥3
result-equivalent positives (jaccard>0.8) at lexical `cos<0.3`, 10 hard negatives.
Held-out test+valid source (0 train leakage). Corpus 3,220 unique queries.

Every anchor–positive pair is surface-divergent, so the **whole-file score is the
adversarial number** (no bins). Reproduce:

```bash
python scripts/eval_rq1_stratified.py --split datasets/wikidata_testset_RQ1_adv500.json \
  --st intfloat/e5-base-v2 --st sentence-transformers/all-mpnet-base-v2 \
  --st sentence-transformers/all-MiniLM-L6-v2 --st microsoft/codebert-base \
  --st microsoft/graphcodebert-base --st razent/spbert-mlm-wso-base \
  --st checkpoints/e5_square_wikidata/best_model \
  --code-llm Qwen/Qwen2.5-Coder-0.5B \
  --sparqlce checkpoints_b128/wikidata/seed42_full/best_model.pt
```

Paper table: `tab-RQ1-Wikidata-adv500.tex`. Best per column **bold** (all SparqlCE).

| Model | MRR@3 | MRR@5 | MRR@10 | Hits@3 | Hits@5 | Hits@10 | nDCG@3 | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **SparqlCE** | **0.071** | **0.085** | **0.104** | **0.104** | **0.164** | **0.302** | **0.047** | **0.056** | **0.076** |
| Qwen2.5-Coder-0.5B | 0.036 | 0.042 | 0.046 | 0.052 | 0.076 | 0.110 | 0.025 | 0.025 | 0.025 |
| CodeBERT | 0.018 | 0.023 | 0.027 | 0.028 | 0.054 | 0.084 | 0.014 | 0.017 | 0.017 |
| E5-Finetuned | 0.016 | 0.021 | 0.027 | 0.020 | 0.044 | 0.090 | 0.010 | 0.014 | 0.016 |
| GraphCodeBERT | 0.013 | 0.021 | 0.024 | 0.022 | 0.054 | 0.082 | 0.011 | 0.016 | 0.016 |
| E5-base-v2 | 0.012 | 0.015 | 0.018 | 0.018 | 0.028 | 0.060 | 0.009 | 0.009 | 0.011 |
| SBERT/MPNet | 0.009 | 0.010 | 0.013 | 0.014 | 0.018 | 0.038 | 0.008 | 0.008 | 0.009 |
| SBERT/MiniLM | 0.009 | 0.010 | 0.012 | 0.012 | 0.018 | 0.032 | 0.007 | 0.007 | 0.007 |
| SPBERT | 0.007 | 0.009 | 0.011 | 0.008 | 0.018 | 0.032 | 0.006 | 0.007 | 0.008 |
| BM25 | 0.005 | 0.005 | 0.006 | 0.008 | 0.010 | 0.014 | 0.005 | 0.005 | 0.005 |
| TF-IDF | 0.004 | 0.004 | 0.004 | 0.006 | 0.006 | 0.008 | 0.004 | 0.004 | 0.004 |

## Reading the result

- **SparqlCE wins every metric at every cutoff — a clean sweep** (bolds all 9
  columns). MRR@10 0.104 is **2.3× the best baseline** (Qwen 0.046) and **17–26×**
  the lexical methods (TF-IDF 0.004 / BM25 0.006). Hits@10 0.302 vs. ≤0.110 for all
  others.
- **The stratified-aggregate confound is gone.** GraphCodeBERT, which edged SparqlCE
  on the stratified MRR average, sits near the floor here (MRR@10 0.024) — its
  earlier lead came entirely from high-lexical-overlap pairs, which this set
  excludes by construction.
- **Backbone ablation holds:** E5-Finetuned (SparqlCE's backbone, fine-tuned) scores
  MRR@10 0.027 — 4× below SparqlCE — so the advantage is the result-aware design.
- Absolute numbers are low by design: this is the hardest possible regime
  (result-equivalent yet lexically disjoint). The **relative** dominance is the
  claim — result-aware retrieval is the only thing that works when surface signal
  is removed.
