# Wikidata RQ1 — randomly-sampled set (representative, ≥3 positives/anchor)

`wikidata_testset_RQ1_rand500.json` — built by
[`scripts/build_random500_wikidata.py`](../scripts/build_random500_wikidata.py)
from the held-out **test** split of `wikidata-triplets-dataset.json`.

## What it is

**500 anchors sampled at random** from the held-out test split, each keeping its
natural result-equivalent positives (≥3) plus **10 hard negatives**. Unlike the
adversarial `adv500` set, there is **no lexical filter** — anchor↔positive
similarity follows its natural distribution, so this is a *representative*
benchmark, not a hard-selected one.

| property | value |
|---|---|
| anchors | 500 (random, seed 42, from 521 eligible) |
| positives / anchor | min 3, mean 10.05, max 20 (5,026 total) |
| positive lexical cos (natural) | min 0.025 / mean 0.369 / max 1.000 |
| — fraction cos<0.3 | 58.5% |
| — fraction cos<0.5 | 79.6% |
| — fraction cos<0.8 | 90.1% |
| positive answer_jaccard | all > 0.8 (result-equivalent) |
| hard negatives / anchor | 10 (lexical cos mean 0.53) |

## Conditions (all verified)

- **Result equivalence:** every positive `answer_jaccard > 0.8`.
- **≥3 positives per anchor.**
- **Random anchors:** deterministic random sample (seed 42) of eligible test
  anchors — no selection by difficulty.
- **Hard negatives:** lexically nearest non-positive queries (result-different).
- **Held-out & leak-free:** 500/500 anchors from the test split; 0 from train, 0
  from valid.

Lexical cos (TF-IDF, word 1–2 grams, fit on source anchors+positives) is reported
per positive but **not used to filter** — the `pos_cos_mean` field records each
anchor's natural mean.

## Schema

Per entry: `anchor_query`; `positive_queries` (list of
`{query, answer_jaccard, lexical_cos}`, ≥3); `negative_queries` (10 ×
`{query, answer_jaccard, neg_type:"hard", lexical_cos}`); `n_positive`,
`pos_cos_mean`.

## How to evaluate

```bash
python scripts/eval_rq1_stratified.py --split datasets/wikidata_testset_RQ1_rand500.json ...
```
There is one implicit population (no cos bins), so the **whole-file aggregate** is
the representative RQ1 number. This set has its own corpus, so absolute scores are
not comparable to the 3,837-corpus main table — use it for cross-model comparison.

## Relation to the other splits
- `adv500` — same size/shape but **cos<0.3** (adversarial, hard-selected).
- `rand500` (this) — **no lexical filter** (representative, random anchors).

Together they bracket the difficulty range: representative vs. worst-case.
