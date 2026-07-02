# Experiment: SQuARE Threshold Sensitivity (reviewer defense)

**Status: not implemented yet — this document is the spec.**

## Why

SQuARE is built on hand-picked thresholds the paper currently states without
justification, and reviewers will push on them (the WSDM assessment explicitly
recommends a "threshold sensitivity mini-table"):

| Threshold | Value | Where used |
|---|---|---|
| Positive: answer-set Jaccard | **> 0.8** | dataset construction + RQ1 correctness rule (`s(q_a, q_p) ≥ 0.8`) |
| Positive: TF-IDF cosine | **< 0.5** (Wikidata) / **< 0.7** (DBpedia) | dataset construction |
| Negative: answer-set Jaccard | **< 0.2** | dataset construction |
| Negative: TF-IDF cosine | **≥ 0.5** | dataset construction |
| Caching hit threshold θ | {0.7, 0.8, 0.9} | already a sweep — no defense needed |

The goal is to show the paper's conclusions do not hinge on the exact values
of 0.8 / 0.2. Three sub-experiments, ordered by cost; A and B are cheap and
should be run first — they may be all that's needed.

## Experiment A — Jaccard distribution is bimodal (cheapest, strongest visual)

**Claim to support:** 0.8 and 0.2 are not arbitrary — they sit in the
low-density valley of the empirical Jaccard distribution, so the labels are
insensitive to the exact cut.

**Procedure:** From `datasets/result_sets_dbpedia.json`, compute pairwise
answer-set Jaccard for all query pairs with non-empty results (1,677 queries
→ ~1.4M pairs; sets pre-converted, runs in minutes). Plot the histogram
(log-scale y). Report what fraction of pairs falls in the "grey zone"
[0.2, 0.8] — if the mass is concentrated near 0 and near 1, one sentence and
a small figure/footnote defuse the objection entirely.

**Output:** `results/threshold_sensitivity/jaccard_histogram_dbpedia.pdf` +
the grey-zone percentage.

## Experiment B — Evaluation-side sensitivity (no retraining)

**Claim to support:** the RQ1 ranking conclusions are stable under the
correctness threshold.

**Procedure:** Re-run the RQ1 evaluation (`06_eval_*` notebooks) with the
correctness rule `s(q_a, q_p) ≥ τ` for τ ∈ {0.6, 0.7, 0.8, 0.9}, using the
**already-trained** models — embeddings are computed once, only the relevance
labels change. Report MRR@10 for SparqlCE and the best baseline at each τ.

**Output:** the 4×2 "threshold sensitivity mini-table" — the deliverable the
assessment asks for. Expected story: absolute numbers shift, the SparqlCE-vs-
baseline gap (≈2.7×) does not.

## Experiment C — Construction-side sensitivity (only if reviewers demand it)

**Claim to support:** training supervision itself is robust to the positive
threshold.

**Procedure:** Rebuild SQuARE positives at Jaccard > {0.7, 0.8, 0.9} (keep
negatives fixed at < 0.2), retrain SparqlCE with the standard recipe
(`notebooks/training/train_dbpedia.ipynb`, seed 42) per variant, evaluate at
the fixed τ = 0.8 rule. Three full training runs — budget a GPU day. Requires
the raw answer-set data used by
`scripts/training/generate_training_triplets.py` to regenerate triplets.

**Recommendation:** run A + B now and put one sentence + the mini-table in the
paper; hold C for the rebuttal unless a co-author disagrees.

## Checklist

- [ ] Notebook `notebooks/evaluation-RQ1/threshold_sensitivity_dbpedia.ipynb` (A + B)
- [ ] Add grey-zone % + mini-table to §Experiments (near the SQuARE construction paragraph)
- [ ] Decide whether C is pre-submission or rebuttal material
