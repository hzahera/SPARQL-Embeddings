# RQ1 — clean natural splits (`{dbpedia,wikidata}_testset_RQ1_clean.json`)

Built by [`scripts/build_clean_rq1.py`](../scripts/build_clean_rq1.py). **One
identical protocol per knowledge graph** — the same flags produce both files.

## Why this set exists

The Wikidata RQ1 story currently rests on splits that are either label-noisy or
difficulty-filtered:

| existing set | problem |
|---|---|
| `wikidata_testset_RQ1.json` (natural) | 40.2% of anchor–positive pairs are **both scalar COUNT aggregates** — `answer_jaccard = 1.0` records that two *numbers* coincided, not that the queries are result-equivalent. A further ~33% share no P-id, no Q-id and no literal with the anchor: equivalence is decided by which entity an opaque external-catalogue id resolves to, a fact about the graph's contents that is absent from the query and unlearnable by any query encoder. |
| `wikidata_testset_RQ1_adversarial.json` | filtered to anchor–positive cosine **< 0.3**. Filtering on lexical similarity selects for *difficulty*, so "SparqlCE wins here" reads as outcome-selected however honestly it was arrived at. Its whole-file aggregate is the `<0.3` stratum by construction. |

This set fixes the **labels** and applies **no lexical filter at all**, so the
natural similarity distribution survives (see the spread printed at build time —
all four cos-bins are populated, unlike the adversarial split which is 100%
`<0.3`). Because the identical rules run on DBpedia too, the Wikidata split
cannot be characterised as tuned: whatever is done to one graph is done to both.

## The two rules (fixed a priori; neither refers to any model's output)

**R1 — non-singleton results.** Anchor and every positive must return
`>= --min-result-size` (default 5) rows. `jaccard > 0.8` between two single-row
results only means "the same one answer", which degenerates to identical (J=1)
or disjoint (J=0) and never tests set overlap. This is the same rationale as
[`wikidata_testset_RQ1_nonsingleton.README.md`](wikidata_testset_RQ1_nonsingleton.README.md);
R1 generalises it to both graphs.

**R2 — no scalar-aggregate collisions.** Drop pairs where *both* queries project
an aggregate with no `GROUP BY` (so each returns one numeric row). R1 removes
most of these via the row count; R2 states the rule explicitly and still applies
where the execution oracle has no entry for a query.

Everything else matches the existing DBpedia RQ1 protocol: positives
`answer_jaccard > 0.8` capped at 20 per anchor (strongest overlap first), the
shipped `jaccard = 0` distractors as negatives, held-out splits only.

## Leakage

Anchors are drawn from `test_idx` (+ `valid_idx` when `--splits test+valid`);
`train_idx` is **never** used. Valid was used only for checkpoint selection —
mild, disclosed optimism that applies equally to SparqlCE and E5-Finetuned, and
is included because the execution oracle covers too few test-only anchors. Same
precedent and same disclosure as the non-singleton set.

## DBpedia (built, committed)

`--kg dbpedia --splits test+valid --min-result-size 5`

| property | value |
|---|---|
| anchors | 152 (test+valid; 0 train) |
| positives/anchor | min 1, mean 8.72, max 20 (1,325 total) |
| result sizes | min 5, median 17, max 2,164 |
| corpus | 501 unique queries |
| lexical spread | `<0.3` 11.8% · `0.3–0.5` 66.4% · `0.5–0.8` 11.2% · `>=0.8` 10.5% |

Oracle coverage is the binding constraint (`result_sets_dbpedia.json` covers
only the 1,677-query RQ1 corpus, so 63.3% of test+valid queries resolve and the
rest are conservatively dropped). Anchor count by setting:

| `--min-result-size` | `--splits test` | `--splits test+valid` |
|---|---|---|
| 2 | 197 | 281 |
| 3 | 144 | 208 |
| **5** (default) | 104 | **152** |
| 10 | 65 | 100 |

152 anchors is a small evaluation — expect wide confidence intervals. If a
paired test against E5-Finetuned comes out underpowered, `--min-result-size 3`
(208 anchors) still breaks the singleton degeneracy and is defensible; report
whichever you pre-commit to, not whichever wins.

## Wikidata (must be built on the cluster)

Needs `result_sets_wikidata.json` (~6 GB), streamed once with `ijson`, so
runtime is dominated by reading that file:

```bash
python scripts/build_clean_rq1.py --kg wikidata --splits test+valid --min-result-size 5
```

## Evaluate — identical command, only `--split` changes

```bash
python scripts/eval_rq1_stratified.py --stratify \
  --split datasets/dbpedia_testset_RQ1_clean.json \
  --st intfloat/e5-base-v2 --st sentence-transformers/all-mpnet-base-v2 \
  --st sentence-transformers/all-MiniLM-L6-v2 --st microsoft/codebert-base \
  --st microsoft/graphcodebert-base --st razent/spbert-mlm-wso-base \
  --st checkpoints/e5_square_dbpedia/best_model \
  --code-llm Qwen/Qwen2.5-Coder-0.5B \
  --sparqlce <retrained-checkpoint>.pt
```

`--stratify` also prints the per-cos-bin breakdown, so the aggregate row and the
`<0.3` stratum come from one run on one unfiltered file — the adversarial split's
claim without the adversarial split's selection.

> **The SparqlCE checkpoint must be retrained.** These splits are unaffected by
> the [`graph_builder.py`](../sparqlce/graph_builder.py) fixes, but existing
> checkpoints were trained on the old representation and will not benefit from
> them. See `RUN_GUIDE.md` Step 1.
