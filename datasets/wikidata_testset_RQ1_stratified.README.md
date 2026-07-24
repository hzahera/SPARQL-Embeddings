# Wikidata RQ1 — stratified, hard-negative test split (Option B)

`wikidata_testset_RQ1_stratified.json` — built by
[`scripts/build_stratified_testset_wikidata.py`](../scripts/build_stratified_testset_wikidata.py)
from `wikidata_testset_RQ1.json` (positives) and `wikidata-triplets-dataset.json`
(jaccard labels + hard-negative mining pool). No new queries are invented.

## What it is

**840 anchors, 210 in each of four lexical-similarity bins**, each paired with
**one** result-equivalent positive (`answer_jaccard > 0.8`; observed min 0.802)
and **20 hard negatives**. It is designed so RQ1 can be reported *per bin* — the
evidence curve where lexical baselines decay as anchor↔positive surface overlap
vanishes while a result-aware method stays flat.

| lexical bin (anchor↔positive TF-IDF cosine) | n | cos min / mean / max |
|---|---:|---|
| `<0.3`   adversarial / surface-divergent | 210 | 0.034 / 0.254 / 0.296 |
| `0.3-0.5` low overlap                     | 210 | 0.319 / 0.420 / 0.499 |
| `0.5-0.8` partial overlap                 | 210 | 0.502 / 0.655 / 0.798 |
| `>=0.8`  near-duplicate (easy contrast)   | 210 | 0.812 / 0.916 / 1.000 |

Equal bins (210 each) give a controlled cross-bin comparison; the `0.5-0.8` bin
(248 eligible anchors) is the capacity bottleneck, so 210/bin is the largest
balanced size. Deterministic (seed 42).

## Why this split exists

The standard RQ1 test set has three confounds that let an aggregate score
reward lexical matching and hide the result-aware advantage:

1. **positives/anchor ranges 1–20** (median 2, 11% capped at 20) → `P@k`/`R@k`
   have hard ceilings and the macro-average mixes incompatible regimes.
2. **`answer_jaccard` is bimodal** (53% of Wikidata result sets are singletons →
   jaccard collapses to 0/1) and **~45% of positives are lexical near-duplicates**
   (cos ≥ 0.3), so the aggregate is dominated by trivially-retrievable pairs.
3. **only 21 distinct negatives**, reused across all anchors → weak distractors.

See [`DATASET_STATS.md`](DATASET_STATS.md) for the full profiling.

This split fixes all three: exactly one positive per anchor (removes the ceiling
artifacts), equal lexical bins (isolates *where* each method wins), and 20 mined
hard negatives per anchor (real distractors).

## Schema

Source schema plus stratification fields:

- `anchor_query`, `positive_queries` (length 1), `negative_queries` (length 20)
- `lexical_bin` ∈ {`<0.3`, `0.3-0.5`, `0.5-0.8`, `>=0.8`}
- `lexical_cos` — anchor↔positive TF-IDF cosine
- `answer_jaccard` — anchor↔positive result overlap (> 0.8)
- each negative: `query`, `answer_jaccard` (null if not in the label LUT),
  `neg_type: "hard"`, `lexical_cos` (anchor↔negative)

## Lexical similarity & bins

Anchor–positive similarity is **TF-IDF cosine over raw SPARQL text (word 1–2
grams)**, fit on the RQ1 test set's anchors+positives — identical basis to
`evaluate_rq1._lexical_cos_fn` and the balanced split, so the bins are directly
comparable across splits. The `0.30` boundary is the adversarial boundary from
the paper's stratified analysis (≈ the pair-distribution median, p50 = 0.288).

## Hard negatives

A hard negative for anchor *a* is a query mined from the full triplets corpus
with high TF-IDF cos(*a*, ·) that is **not** one of *a*'s result-equivalent
positives (and, where a jaccard label exists for the pair, has jaccard ≤ 0.8).
Lexically similar but not result-equivalent — so only result/structure signal can
reject them. Observed hard-negative lexical cos: mean 0.49 / median 0.46 (vs. the
old shared negatives, which were lexically random). Result-difference is verified
against the answer-set labels where available (21 pairs had a label; none exceeded
0.8); the remainder rely on the dataset's own positive labeling being the complete
result-equivalent set for each anchor.

## How to evaluate

Point the RQ1 notebook / `evaluate_rq1.py` at this file as for any test set, then
**report MRR/Hits per `lexical_bin`** (group the per-anchor results by the field).
The headline RQ1 claim is the shape of that curve: lexical/text methods fall off
from `>=0.8` down to `<0.3`; the result-aware method stays flat.

⚠️ **Caveat.** Like the balanced split, this file has its own corpus, so absolute
scores are **not** comparable to the 3837-corpus numbers in the main RQ1 table —
use it for *cross-model* and *cross-bin* comparison. And do **not** re-run the
notebook's `report_stratified()` on this file: it re-fits TF-IDF on the subset, so
its cosines drift from the fixed build-time bins. The `lexical_bin` field is the
authority.
