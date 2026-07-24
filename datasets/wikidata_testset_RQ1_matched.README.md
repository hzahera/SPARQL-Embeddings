# Wikidata RQ1 — DBpedia-matched test split

`wikidata_testset_RQ1_matched.json` — built by
[`scripts/build_dbpedia_matched_wikidata.py`](../scripts/build_dbpedia_matched_wikidata.py)
from the held-out **test** split of `wikidata-triplets-dataset.json`. Designed to
mirror `dbpedia_testset_RQ1.json` on every construction parameter, so Wikidata and
DBpedia RQ1 results are reported on structurally identical test sets.

## Match to DBpedia

| Parameter | DBpedia (`dbpedia_testset_RQ1.json`) | Wikidata (this) |
|---|---|---|
| anchors | 462 | **462** |
| positives/anchor (mean / median / max) | 8.18 / 4 / 20 | 7.94 / 3 / 20 |
| positives histogram (1 / 2–4 / 5–9 / 10–19 / 20) | 165 / 83 / 53 / 34 / 127 | 165 / 83 / 53 / 39 / 122 |
| negatives/anchor | 20 | 20 |
| positive threshold | jaccard > 0.8 (min 0.804) | jaccard > 0.8 (min 0.802) |
| negative threshold | jaccard = 0 | jaccard = 0 |
| corpus (unique queries) | 1,841 | 1,968 |

The only deviation: the 20-positive (capped) bucket had 122 eligible Wikidata
anchors vs. DBpedia's 127, so 5 anchors spilled into the 10–19 bucket. This nudges
the mean to 7.94 (vs. 8.18) — otherwise the profiles are identical.

## Construction

- **Source:** held-out Wikidata TEST split only (verified: 462/462 from test, 0
  from train/valid — no leakage).
- **Positives:** all result-equivalent (`answer_jaccard > 0.8`) positives per
  anchor, capped at 20 (highest-jaccard kept when >20).
- **Negatives:** the anchor's 20 shipped negatives (`answer_jaccard = 0`); Wikidata
  uses ~21 distinct shared negatives, matching DBpedia's ~23.
- **Anchor sampling:** stratified to reproduce DBpedia's positives-per-anchor
  histogram; deterministic (seed 42).

## Schema

Per entry: `anchor_query`; `positive_queries` (list of `{query, answer_jaccard}`,
1–20); `negative_queries` (20 × `{query, answer_jaccard:0.0}`).

## How to evaluate

```bash
python scripts/eval_rq1_stratified.py --split datasets/wikidata_testset_RQ1_matched.json ...
```
No lexical bins — this is the natural-distribution analogue of the DBpedia table.
Report the whole-file aggregate, directly comparable in structure to DBpedia RQ1.
Note it has its own corpus, so absolute scores are cross-model comparable within
Wikidata, not against DBpedia's 1,841-corpus numbers.
