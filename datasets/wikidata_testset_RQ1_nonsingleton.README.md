# Wikidata RQ1 — non-singleton (substantive result-equivalence) test set

`wikidata_testset_RQ1_nonsingleton.json` — built by
[`scripts/build_nonsingleton_wikidata.py`](../scripts/build_nonsingleton_wikidata.py)
from the held-out Wikidata **test** split.

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
| anchors | 215 (held-out test only, 0 leakage) |
| positives/anchor | min 1, mean 3.26, max 18 (700 total) |
| positive answer_jaccard | all > 0.8 (min 0.802) |
| result-set size (anchor + positives) | min 5, median 94, max 2,018,536 (all ≥ 5) |
| negatives/anchor | 20 (shipped, jaccard 0) |
| corpus | 718 unique queries |

Each record carries `anchor_result_size` and per-positive `result_size` for
audit. Negatives are the shipped `jaccard = 0` distractors from
`wikidata_testset_RQ1.json`; they are absent from the cached oracle and, being
disjoint, never contributed to the bimodality, so they are not size-filtered.

## Construction (leak-free, deterministic)

* Source: held-out Wikidata `test_idx` only (no train/val → 0 leakage).
* Anchor kept iff its result set has ≥ 5 rows.
* Positive kept iff `answer_jaccard > 0.8` **and** result set ≥ 5 rows, capped
  at 20/anchor (strongest overlap first).
* Anchor kept iff it retains ≥ 1 such positive.
* Result-set sizes come from the cached execution oracle
  `result_sets_wikidata.json`, streamed once with `ijson`. Queries absent from
  the oracle are treated as singletons (size 0 → filtered), i.e. conservative.

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
