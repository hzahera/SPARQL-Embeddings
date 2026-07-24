# Wikidata RQ1 — adversarial (cos < 0.3), result-equivalent test split

`wikidata_testset_RQ1_balanced.json` — built by
[`scripts/build_balanced_testset_wikidata.py`](../scripts/build_balanced_testset_wikidata.py)
from `wikidata_testset_RQ1.json` (no new queries; a filtered subset).

## What it is

471 anchors, each paired with **one** result-equivalent positive
(`answer_jaccard > 0.8`; observed min 0.837) whose anchor–positive lexical
similarity is **below 0.30**. This is the purely adversarial regime: anchor and
its result-equivalent positive share almost no surface form, so lexical
(TF-IDF/BM25) and text-embedding methods have essentially no signal, and only
result-aware / structural matching can retrieve.

- lexical cos: min 0.025 / mean 0.224 / max 0.296
- corpus: 805 unique queries
- 471 is the natural maximum at one-positive-per-anchor (only 471 test anchors
  have a result-equivalent positive with cos < 0.30); reaching 500 would require
  admitting multiple positives per anchor, which is avoided (see below).

Each entry keeps the source schema (`anchor_query`, `positive_queries`,
`negative_queries`) plus `lexical_bin` (always `low`), `lexical_cos`,
`answer_jaccard`.

## Why cos < 0.3

Anchor–positive lexical similarity is **TF-IDF cosine over the raw SPARQL text
(word 1–2 grams)** — identical to `evaluate_rq1._lexical_cos_fn`. `0.30` is the
adversarial boundary already established in the paper's stratified analysis and
sits at ≈ the median of the pair distribution (p50 = 0.288). Below it, shared
vocabulary/bigrams are negligible; this is the setting the result-aware method
is designed for.

## Construction choices

- **One positive per anchor** — the hardest (lowest-cos) qualifying positive —
  so an anchor never appears twice and its other positives never become false
  distractors in the shared corpus.
- Every eligible anchor included (no subsampling). Deterministic (seed 42).

## How to evaluate

Point the RQ1 notebook / `evaluate_rq1.py` at this file exactly as for the
standard test set. The **whole-file aggregate** MRR/Hits/nDCG *is* the `cos<0.3`
result — no per-bin filtering needed.

Do **not** read the notebook's `report_stratified()` block on this file: it
re-fits TF-IDF on this 471-anchor subset, so its cosines drift upward and its
bin counts will not match the fixed build-time `cos<0.3` selection. The aggregate
over the whole file is the number to report.

⚠️ **Caveat.** This split has its own corpus (805 unique queries), so absolute
scores here are **not** comparable to the 3837-corpus numbers in the main RQ1
table. Use it for **cross-model** comparison on a common footing.
