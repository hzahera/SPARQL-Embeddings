# Wikidata RQ1 — adversarial (low-lexical-similarity) test set

`wikidata_testset_RQ1_adversarial.json` — built by
[`scripts/build_adversarial_wikidata.py`](../scripts/build_adversarial_wikidata.py)
from the held-out Wikidata **test** split. Companion to the natural-distribution
`wikidata_testset_RQ1.json`; together they bracket the difficulty range.

## What it is

**553 anchors**, each paired only with result-equivalent positives
(`answer_jaccard > 0.8`) whose lexical similarity to the anchor is **< 0.30**, plus
the shipped 20 negatives (`answer_jaccard = 0`). This is the regime that tests the
paper's thesis: with no surface signal, only result-aware / structural matching can
retrieve.

| property | value |
|---|---|
| anchors | 553 (held-out test only, 0 leakage) |
| positives/anchor | min 1, mean 7.51, max 20 (4,153 total) |
| positive lexical cos (expanded) | min 0.005 / mean 0.160 / **max 0.300** |
| positive answer_jaccard | min 0.804 (all > 0.8) |
| negatives/anchor | 20 (shipped, jaccard 0) |
| corpus | 2,204 unique queries |

## Critical: similarity is measured on label-EXPANDED text

Lexical cosine is TF-IDF over the **label-expanded** query text
(`wdt:P31 → instance of (P31)`), i.e. the *same* text the text baselines read at
eval time after the fairness fix — not the raw opaque-id text. A pair can be
cos<0.3 on raw `wdt:P31 wd:Q5` yet share tokens once expanded; filtering on raw
text would leave the baselines residual signal and overstate the adversarial gap.
Filtering on expanded text guarantees the baselines are genuinely blind. The
expansion is byte-identical to `sparqlce.sparql_utils.expand_wikidata_labels`.

## Result (fair; all text baselines label-expanded)

Whole-file aggregate = the adversarial number (every positive is cos<0.3). See
`tab-RQ1-Wikidata-adversarial.tex`. SparqlCE wins every metric at every cutoff:

| Model | MRR@10 | Hits@10 | nDCG@10 |
|---|---:|---:|---:|
| **SparqlCE** | **0.218** | **0.443** | **0.201** |
| E5-Finetuned | 0.198 | 0.378 | 0.187 |
| SBERT/MPNet | 0.107 | 0.190 | 0.086 |
| E5-base-v2 | 0.097 | 0.193 | 0.079 |
| SBERT/MiniLM | 0.097 | 0.199 | 0.078 |
| Qwen2.5-Coder-0.5B | 0.077 | 0.172 | 0.057 |
| CodeBERT | 0.066 | 0.143 | 0.049 |
| GraphCodeBERT | 0.062 | 0.137 | 0.050 |
| SPBERT | 0.057 | 0.134 | 0.042 |
| BM25 | 0.040 | 0.110 | 0.036 |
| TF-IDF | 0.029 | 0.085 | 0.029 |

**SparqlCE beats its own fine-tuned backbone** (E5-Finetuned): +10% MRR@10, +17%
Hits@10 — the graph/structural component pays off exactly where lexical similarity
vanishes. On the natural split (`wikidata_testset_RQ1.json`) the ordering reverses
(E5-Finetuned leads), which is the point: structure matters in the adversarial
regime, not when surface signal already suffices.

## How to reproduce

```bash
python scripts/build_adversarial_wikidata.py        # needs datasets/wikidata_labels.json
python scripts/eval_rq1_stratified.py --split datasets/wikidata_testset_RQ1_adversarial.json \
  --st intfloat/e5-base-v2 --st sentence-transformers/all-mpnet-base-v2 \
  --st sentence-transformers/all-MiniLM-L6-v2 --st microsoft/codebert-base \
  --st microsoft/graphcodebert-base --st razent/spbert-mlm-wso-base \
  --st checkpoints/e5_square_wikidata/best_model \
  --code-llm Qwen/Qwen2.5-Coder-0.5B \
  --sparqlce checkpoints_b128/wikidata/seed42_full/best_model.pt
```
