# Wikidata RQ1 — non-singleton (substantive result-equivalence) test set

`wikidata_testset_RQ1_nonsingleton.json` — built by
[`scripts/build_nonsingleton_wikidata.py`](../scripts/build_nonsingleton_wikidata.py)
from the held-out Wikidata **test + valid** splits.

## Why this set exists

Wikidata's `answer_jaccard` is bimodal (0 or 1): most queries return a single
entity, so two singletons are either identical (jaccard 1) or disjoint
(jaccard 0). Under that degeneracy `jaccard > 0.8` just means "same one answer"
and never tests partial overlap. This set removes the degeneracy by requiring
**both the anchor and every positive to return at least 5 rows**, so a
`jaccard > 0.8` positive is a genuine many-element set intersection. It also
drops every `COUNT(...)` single-row aggregate. **No lexical filter** is applied —
this is the natural lexical distribution over non-singleton, result-equivalent
pairs.

## What it is

| property | value |
|---|---|
| anchors | 275 (215 test + 60 valid; 0 train) |
| positives/anchor | min 1, mean 3.57, max 18 (981 total) |
| positive answer_jaccard | all > 0.8 (min 0.802) |
| result-set size (anchor + positives) | min 5, median 69, max 2,018,536 (all ≥ 5) |
| negatives/anchor | 20 (shipped, jaccard 0) |
| corpus | 773 unique queries |

Each record carries `anchor_result_size` and per-positive `result_size` for
audit. Negatives are the shipped `jaccard = 0` distractors; they are absent from
the cached oracle and, being disjoint, never contributed to the bimodality, so
they are not size-filtered.

## Leakage

**No training leakage.** Anchors are drawn only from `test_idx` + `valid_idx`;
`train_idx` is excluded, so neither trained model (SparqlCE, E5-Finetuned) ever
saw these anchors as a training target. Valid was used only for checkpoint
selection (mild optimism) and is included because the cached execution oracle
covers too few *test-only* anchors to form a large non-singleton set (215 test
alone; +60 from valid → 275). Report this as a held-out test+valid evaluation.

## Construction (leak-free, deterministic)

* Source: held-out Wikidata `test_idx` + `valid_idx` (`SPLIT_KEYS`), train excluded.
* Anchor kept iff its result set has ≥ 5 rows.
* Positive kept iff `answer_jaccard > 0.8` **and** result set ≥ 5 rows, capped
  at 20/anchor (strongest overlap first).
* Anchor kept iff it retains ≥ 1 such positive.
* Result-set sizes come from the cached execution oracle
  `result_sets_wikidata.json`, streamed once with `ijson`. Queries absent from
  the oracle are treated as singletons (size 0 → filtered), i.e. conservative.

## Note on the evaluation metric (not result-aware — but it doesn't matter here)

`scripts/eval_rq1_stratified.py` scores relevance as binary membership in each
anchor's labeled `positive_queries` list, not by recomputing answer-set Jaccard
at rank time. We verified this is not a confound for this set: recomputing true
result-equivalence (jaccard>0.8, entity-normalized) against the oracle over the
full 773-query corpus finds only **33 unlabeled result-equivalents across all
275 anchors** — because the positives were themselves built from jaccard>0.8, the
label set already *is* the result-aware relevance set. A result-aware metric
would not change the ranking on this set.

## How to reproduce

```bash
python scripts/build_nonsingleton_wikidata.py        # needs datasets/result_sets_wikidata.json

python scripts/eval_rq1_stratified.py \
  --split datasets/wikidata_testset_RQ1_nonsingleton.json \
  --st intfloat/e5-base-v2 --st sentence-transformers/all-mpnet-base-v2 \
  --st sentence-transformers/all-MiniLM-L6-v2 --st microsoft/codebert-base \
  --st microsoft/graphcodebert-base --st razent/spbert-mlm-wso-base \
  --st checkpoints/e5_square_wikidata/best_model \
  --code-llm Qwen/Qwen2.5-Coder-0.5B \
  --sparqlce checkpoints_b128/wikidata/seed42_full/best_model.pt
```

All *text* baselines receive the same Wikidata label-expanded query text
SparqlCE resolves internally (`expand_wikidata_labels`), for a fair comparison.
The missing `lexical_bin` field defaults to the whole-file "all" aggregate.
