# Wikidata RQ1 — lexically-balanced, result-equivalent test split

`wikidata_testset_RQ1_balanced.json` — built by
[`scripts/build_balanced_testset_wikidata.py`](../scripts/build_balanced_testset_wikidata.py)
from `wikidata_testset_RQ1.json` (no new queries; a re-balanced subset).

## What it is

788 anchors, each paired with **one** result-equivalent positive
(`answer_jaccard > 0.8`; observed min 0.802), balanced 50/50 across two
anchor–positive lexical-similarity bins. Near-duplicate pairs (`cos >= 0.80`)
are **dropped**.

| bin | lexical cos | n | share | observed cos (min / mean / max) |
|-----|-------------|----|-------|---------------------------------|
| low | `< 0.30`        | 394 | 50% | 0.034 / 0.242 / 0.296 |
| mid | `0.30 – 0.80`   | 394 | 50% | 0.319 / 0.547 / 0.798 |

Each entry keeps the source schema (`anchor_query`, `positive_queries`,
`negative_queries`) plus `lexical_bin`, `lexical_cos`, `answer_jaccard`
(the evaluation pipeline ignores the extra keys; use them to filter by bin).

## Why these thresholds

Anchor–positive lexical similarity is **TF-IDF cosine over the raw SPARQL text
(word 1–2 grams)** — identical to `evaluate_rq1._lexical_cos_fn`. Raw (not
label-expanded) text is used so the bins mean exactly what they mean in the
paper's existing stratified analysis, keeping this split comparable to it.

- **low, `cos < 0.30` — surface-divergent / adversarial.** Anchor and its
  result-equivalent positive share almost no vocabulary or bigrams, so lexical
  (TF-IDF/BM25) and text-embedding signals are near-random; only result-aware /
  structural matching can retrieve. `0.30` is the boundary already established
  in the paper's stratified analysis (the `<0.3` bin) and sits at ≈ the median
  of the pair distribution (p50 = 0.288).
- **mid, `0.30 – 0.80`** — partial overlap; the transition regime where methods
  begin to differentiate.
- **dropped, `cos >= 0.80` — near-duplicate.** Anchor and positive are almost
  identical in surface form; retrieval is trivial for every method, so these
  pairs carry no discriminative signal and are removed. This focuses the
  benchmark on the regimes where methods actually differ.

**Composition (50 / 50).** Equal weight on the two discriminative regimes so
per-bin metrics have equal, adequate sample sizes and the aggregate is not
swamped by easy cases. The low bin (471 eligible) and mid bin (421 eligible)
support up to 394 anchors each (the shared-eligibility constraint fixes the
maximum), giving N = 788.

## Construction choices

- **One positive per anchor.** An anchor never appears twice, so its other
  positives never become false distractors in the shared corpus (which would
  wrongly penalise MRR). Representative positive per (anchor, bin): lowest cos
  for `low` (hardest), nearest the bin centre for `mid`.
- **Greedy scarce-bin-first assignment** (seed 42) hits the 50/50 target.

## How to evaluate

Point the RQ1 notebook / `evaluate_rq1.py` at this file exactly as for the
standard test set — all models read label-expanded queries via their transforms;
SparqlCE expands internally.

- **Aggregate** on the whole file reports MRR/Hits/nDCG under the balanced
  50/50 lexical regime.
- **Per bin**, filter entries by the `lexical_bin` field and run `report()` on
  each subset (against the shared corpus) to get full **MRR/Hits/nDCG** per bin.
  Do **not** rely on the notebook's built-in `report_stratified()` here: it
  re-fits TF-IDF on this subset, so its cosines (and therefore its bin counts)
  differ from the fixed `lexical_bin` labels stored at build time. The stored
  `lexical_bin` is authoritative.

⚠️ **Caveat.** This split has its own corpus (1397 unique queries) and its own
balanced anchor distribution, so absolute scores here are **not** comparable to
the 3837-corpus numbers in the main RQ1 table. Use it for **cross-model and
cross-bin** comparison on a common footing, not as a drop-in replacement for the
main aggregate.
