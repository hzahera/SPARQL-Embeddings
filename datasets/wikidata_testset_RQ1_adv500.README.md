# Wikidata RQ1 — purely adversarial set (cos<0.3, ≥3 positives/anchor)

`wikidata_testset_RQ1_adv500.json` — built by
[`scripts/build_adversarial500_wikidata.py`](../scripts/build_adversarial500_wikidata.py)
from the held-out **test + validation** splits of `wikidata-triplets-dataset.json`.
No new queries are invented.

## What it is

**500 anchors**, each paired with **≥3 result-equivalent positives** whose lexical
overlap with the anchor is negligible, plus **10 hard negatives**. This is the
paper's core claim in one dataset: anchor and its positives share almost no surface
form, so lexical (TF-IDF/BM25) and text-embedding methods have essentially no
signal — only result-aware / structural matching can retrieve.

| property | value |
|---|---|
| anchors | 500 |
| positives / anchor | min 3, mean 9.65, max 20 (4,823 total) |
| positive lexical cos (anchor↔positive) | min 0.012 / mean 0.206 / **max 0.300** |
| positive answer_jaccard | min 0.804 (all > 0.8) |
| hard negatives / anchor | 10 |
| hard-negative lexical cos to anchor | mean 0.49 |
| unique corpus queries | 2,831 |

## Conditions (all enforced, verified)

- **Lexical divergence:** every anchor↔positive TF-IDF cosine `< 0.30`.
- **Result equivalence:** every positive `answer_jaccard > 0.8`.
- **≥3 positives per anchor** (multi-positive, unlike the one-positive balanced/
  stratified splits).
- **Hard negatives:** lexically nearest queries (mined from the full triplets
  corpus) that are **not** result-equivalent positives of the anchor; where a
  jaccard label exists for the pair it is `≤ 0.8` (observed max 0.0).

Lexical similarity is TF-IDF over raw SPARQL text (word 1–2 grams), fit on the
source anchors+positives — identical basis to `evaluate_rq1._lexical_cos_fn` and
the other RQ1 splits, so `cos < 0.3` means exactly what it means elsewhere.

## Source & leakage

Anchors are drawn **only from the held-out test + validation splits** (train
excluded). Verified: 500/500 anchors are in test∪valid, **0 leak from train**.
The validation split is used only for model selection in the pipeline, so it is
safe as evaluation data. 625 anchors were eligible (≥3 adversarial positives);
500 were sampled deterministically (seed 42).

> If your SparqlCE checkpoint was trained on *all* splits rather than train-only,
> re-point the builder at `test_idx` alone (max ~305 anchors) to stay strictly
> held-out.

## Schema

Per entry: `anchor_query`; `positive_queries` (list of
`{query, answer_jaccard, lexical_cos}`, length ≥3); `negative_queries` (list of
`{query, answer_jaccard, neg_type:"hard", lexical_cos}`, length 10); plus
`lexical_bin:"low"`, `n_positive`, `pos_cos_mean`, `pos_jaccard_min`.

## How to evaluate

Point `evaluate_rq1.py` / `scripts/eval_rq1_stratified.py` at this file. Because
**every** anchor↔positive pair is `cos < 0.3`, the **whole-file aggregate**
MRR/Hits/nDCG *is* the adversarial number — no per-bin stratification needed.

⚠️ This file has its own corpus (2,831 queries), so absolute scores are not
comparable to the 3,837-corpus main RQ1 table — use it for cross-model comparison
on a common, purely-adversarial footing.
