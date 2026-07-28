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

**Group-level disjointness is not pair-level disjointness.** The above was the
only leakage claim this file made, and it is not sufficient. `splits_{kg}.json`
partitions anchor *groups*, but the same SPARQL query text recurs across groups,
so a held-out group can restate an anchor–positive pair that was optimized over
during training. Measured on the first builds of these files:

| split | eval positive pairs also trained on | anchors with ≥1 |
|---|---|---|
| `wikidata_testset_RQ1_clean.json` | 1,635 / 1,935 (84.5%) | 424 / 450 (94.2%) |
| `dbpedia_testset_RQ1_clean.json` | 1,116 / 1,325 (84.2%) | 144 / 152 (94.7%) |

(vs `train_idx` alone, excluding valid: Wikidata 1,448/1,935 = 74.8%.)

Both models train on the same pairs, so this is **not** an unfairness between
them — but it does not help them equally, which makes it decisive for the
headline number. On the 5-seed Wikidata sweep:

| MRR@10 | leaked anchors | non-leaked | all |
|---|---|---|---|
| SparqlCE | 0.795 | 0.577 | 0.770 |
| E5-Finetuned | 0.475 | 0.521 | 0.480 |
| **gap** | **+0.321** | **+0.057** | +0.290 |

Roughly 85% of the reported gap comes from restated training pairs. **R3
(`--leak-filter`, on by default) removes them**, producing the `_noleak` files
that everything downstream should use:

| split | anchors | positives |
|---|---|---|
| `wikidata_testset_RQ1_clean_noleak.json` | 136 | 300 |
| `dbpedia_testset_RQ1_clean_noleak.json` | 51 | 209 |

Audit any split (and break exported scores down by leaked/non-leaked) with
[`scripts/check_leakage.py`](../scripts/check_leakage.py); `run_seed_sweep.sh`
runs it as a pre-flight and refuses a leaky split unless `ALLOW_LEAKY_SPLIT=1`.
The smaller anchor counts are the honest measurable subset — a 136-anchor
uncontaminated result supports a claim that a 450-anchor contaminated one does
not.

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

## Wikidata — use R2 only (do **not** apply R1 here)

```bash
python scripts/build_clean_rq1.py --kg wikidata --min-result-size 0 \
  --from-split datasets/wikidata_testset_RQ1_stratified.json \
  --out datasets/wikidata_testset_RQ1_clean.json
```

**450 anchors, 1,923 corpus queries, no execution oracle needed**, natural
lexical spread across all four bins:

| | anchors | `<0.3` | `0.3–0.5` | `0.5–0.8` | `>=0.8` |
|---|---|---|---|---|---|
| **R2 only (recommended)** | **450** | 118 (26.2%) | 100 (22.2%) | 84 (18.7%) | 148 (32.9%) |
| R1+R2, `min-result-size 3` | 291 | **2 (0.7%)** | 82 | 84 | 123 |

### Why R1 must be disabled on Wikidata

Two independent failures, both visible in the build output:

1. **The oracle does not cover the splits.** `result_sets_wikidata.json` resolves
   only **37.9%** (2,229/5,881) of test+valid queries — 2,229 is exactly the
   Wikidata corpus size in `SUPPLEMENTARY.md` §3. The file is ~6 GB because a few
   thousand queries return millions of rows each, not because it is broad. So
   `anchor_small` drops are overwhelmingly *unknown* size, not *verified small*:
   R1 ends up selecting for oracle presence. The builder now prints a WARNING
   whenever coverage is below 75%.
2. **R1 is a difficulty filter on Wikidata, pointing the wrong way.** The
   multi-row Wikidata queries *are* the `BIND` / property-path template families,
   which are lexical near-duplicates of one another. Requiring ≥3 rows therefore
   selects *toward* high similarity (mean cos 0.704, 42.3% in `>=0.8`, 2 anchors
   in `<0.3`) — the regime where a strong text encoder already wins. R1 was
   designed to remove label degeneracy on DBpedia; on Wikidata it removes the
   interesting half of the distribution.

R2 needs no oracle and is unaffected by both problems. It removes 49
scalar-aggregate anchors and 188 scalar-aggregate positives — the pairs whose
`answer_jaccard = 1.0` only records that two numbers coincided.

R1 remains correct and enabled by default for DBpedia, where the corpus has
**0.0%** aggregates and no BIND/path template families.

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
