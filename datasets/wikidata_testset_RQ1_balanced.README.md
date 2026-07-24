# Wikidata RQ1 — lexically-balanced, result-equivalent test split

`wikidata_testset_RQ1_balanced.json` — built by
[`scripts/build_balanced_testset_wikidata.py`](../scripts/build_balanced_testset_wikidata.py)
from `wikidata_testset_RQ1.json` (no new queries; a re-balanced subset).

## What it is

761 anchors, each paired with **one** result-equivalent positive
(`answer_jaccard > 0.8`; observed min 0.802), re-balanced across three
anchor–positive lexical-similarity regimes:

| bin | lexical cos | n | share | observed cos (min / mean / max) |
|-----|-------------|----|-------|---------------------------------|
| low    | `< 0.30`        | 304 | 40% | 0.034 / 0.253 / 0.296 |
| medium | `0.30 – 0.60`   | 304 | 40% | 0.319 / 0.460 / 0.599 |
| high   | `>= 0.60`       | 153 | 20% | 0.603 / 0.881 / 1.000 |

Each entry keeps the source schema (`anchor_query`, `positive_queries`,
`negative_queries`) plus `lexical_bin`, `lexical_cos`, `answer_jaccard`
(the evaluation pipeline ignores the extra keys; use them to filter by bin).

## Why these thresholds

Anchor–positive lexical similarity is **TF-IDF cosine over the raw SPARQL
text (word 1–2 grams)** — identical to `evaluate_rq1._lexical_cos_fn`. Raw
(not label-expanded) text is used so the bins mean exactly what they mean in
the paper's existing stratified analysis, keeping this split comparable to it.

- **low, `cos < 0.30` — surface-divergent / adversarial.** Anchor and its
  result-equivalent positive share almost no vocabulary or bigrams, so lexical
  (TF-IDF/BM25) and text-embedding signals are near-random; only result-aware /
  structural matching can retrieve. `0.30` is the boundary already established
  in the paper's stratified analysis (the `<0.3` bin) and sits at ≈ the median
  of the pair distribution (p50 = 0.288), cleanly separating the lower half.
- **high, `cos >= 0.60` — near-paraphrase (≈ p80).** Dominant shared surface
  form (predicates, entities, structure); text encoders retrieve these
  trivially. Kept as a smaller **control** regime that confirms the benchmark is
  not degenerate — every method, including the text baselines, should score well
  here.
- **medium, `0.30 – 0.60`** — partial overlap; the transition regime where
  methods begin to differentiate.

**Composition (40 / 40 / 20).** Low+medium (80%) deliberately over-represents
the regimes that stress result-aware retrieval relative to their natural
frequency, so per-bin metrics have adequate sample sizes and the aggregate is
not swamped by easy near-duplicates. The 20% high control keeps a sanity check
that surface form, when present, is still exploited. The medium bin is the
binding constraint (304 anchors have a positive in `[0.30, 0.60)`), which fixes
the split size at 761 for an exact 40/40/20 balance.

## Construction choices

- **One positive per anchor.** An anchor never appears twice, so its other
  positives never become false distractors in the shared corpus (which would
  wrongly penalise MRR). Representative positive per (anchor, bin): lowest cos
  for `low` (hardest), highest for `high` (easiest), nearest the bin centre for
  `medium`.
- **Greedy scarce-bin-first assignment** (seed 42) hits the target proportions.

## How to evaluate

Point the RQ1 notebook / `evaluate_rq1.py` at this file exactly as for the
standard test set — all models read label-expanded queries via their transforms;
SparqlCE expands internally.

- **Aggregate** on the whole file reports MRR/Hits/nDCG under the balanced
  40/40/20 lexical regime.
- **Per bin**, filter entries by `lexical_bin` and run `report()` on each
  subset (against the shared corpus) to get full **MRR/Hits/nDCG** per bin —
  unlike `report_stratified()`, which returns MRR only.

⚠️ **Caveat.** This split has its own corpus (1387 unique queries) and its own
balanced anchor distribution, so absolute scores here are **not** comparable to
the 3837-corpus numbers in the main RQ1 table. Use it for **cross-model and
cross-bin** comparison on a common footing, not as a drop-in replacement for the
main aggregate.
