# Wikidata RQ1 — surface-divergent, result-equivalent test split

`wikidata_testset_RQ1_balanced.json` — built by
[`scripts/build_balanced_testset_wikidata.py`](../scripts/build_balanced_testset_wikidata.py)
from `wikidata_testset_RQ1.json` (no new queries; a filtered subset).

## What it is

616 anchors, each paired with **one** result-equivalent positive
(`answer_jaccard > 0.8`; observed min 0.802) whose anchor–positive lexical
similarity is **below 0.5**. This isolates the surface-divergent regime — the
one that motivates result-aware retrieval — by removing every pair where the
query text is even moderately similar.

A `lexical_bin` sub-tag is kept so the set can still be split finer:

| sub-bin | lexical cos | n | share | observed cos (min / mean / max) |
|---------|-------------|----|-------|---------------------------------|
| low | `< 0.30`      | 471 | 76% | 0.025 / 0.224 / 0.296 |
| mid | `0.30 – 0.50` | 145 | 24% | 0.319 / 0.441 / 0.499 |

Each entry keeps the source schema (`anchor_query`, `positive_queries`,
`negative_queries`) plus `lexical_bin`, `lexical_cos`, `answer_jaccard`.

## Why cos < 0.5

Anchor–positive lexical similarity is **TF-IDF cosine over the raw SPARQL text
(word 1–2 grams)** — identical to `evaluate_rq1._lexical_cos_fn`. Raw (not
label-expanded) text is used so the threshold means the same thing as in the
paper's existing stratified analysis.

`cos < 0.5` keeps only pairs whose queries share little to no surface form. In
this regime lexical (TF-IDF/BM25) and text-embedding methods have little signal
to exploit, so it is the setting where result-aware / structural matching can
help and where the methods actually separate. Pairs with `cos >= 0.5` (partial
to near-duplicate overlap) are trivially retrievable by text methods and are
excluded. `0.30` (the `low`/`mid` sub-boundary) is the adversarial boundary
already established in the paper's stratified analysis (≈ the median of the pair
distribution, p50 = 0.288).

Every eligible anchor is included (no subsampling); the split is not
proportion-balanced, so it reflects the natural 76/24 split of the divergent
regime. Set `BALANCE = True` in the builder to subsample to fixed proportions
instead.

## Construction choices

- **One positive per anchor** — the hardest (lowest-cos) qualifying positive —
  so an anchor never appears twice and its other positives never become false
  distractors in the shared corpus (which would wrongly penalise MRR).
- Deterministic (seed 42).

## How to evaluate

Point the RQ1 notebook / `evaluate_rq1.py` at this file exactly as for the
standard test set — all models read label-expanded queries via their transforms;
SparqlCE expands internally.

- **Aggregate** on the whole file reports MRR/Hits/nDCG for the full
  surface-divergent (`cos < 0.5`) regime.
- **Per sub-bin**, filter entries by the `lexical_bin` field and run `report()`
  on each subset (against the shared corpus) for full MRR/Hits/nDCG. Do **not**
  rely on the notebook's `report_stratified()` here: it re-fits TF-IDF on this
  subset, so its cosines and bin counts differ from the fixed `lexical_bin`
  labels stored at build time. The stored `lexical_bin` is authoritative.

⚠️ **Caveat.** This split has its own corpus (1070 unique queries), so absolute
scores here are **not** comparable to the 3837-corpus numbers in the main RQ1
table. Use it for **cross-model** comparison on a common footing.
