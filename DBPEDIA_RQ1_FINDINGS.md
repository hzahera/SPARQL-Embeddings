# DBpedia RQ1 — findings

What the DBpedia retrieval experiments established, why the earlier numbers are
not usable, and what the corpus can and cannot support. Every figure here is
reproducible from the scripts named beside it.

**Headline.** On DBpedia, SparqlCE and a supervision-matched fine-tuned E5
control are **statistically indistinguishable** at equal embedding width, equal
training data and equal negatives, over five seeds. The reason is a measured
property of the corpus rather than of the method, and two independent attempts
to build a corpus that *would* separate them both failed in ways the same
measurement predicts.

---

## 1. The benchmark finding

`scripts/build_component_splits.py`

The SQuARE-DBpedia corpus is **not 4,713 independent training items**. It is one
symmetric result-equivalence relation over a closed set of 4,713 queries:

| measurement | value |
|---|---|
| anchor groups | 4,713 |
| positive pairs | 39,520 |
| positives that are themselves anchors | **39,520 / 39,520 (100%)** |
| non-anchor queries in the whole corpus | 14 |
| distinct negative queries in the whole corpus | **24** |

Two consequences follow directly.

**A group-level train/test split cannot be leak-free.** Query *q* is an anchor in
a training group and a positive of a held-out anchor *a*, so the pair `{a, q}`
is restated at evaluation time. This is the origin of the 84.2% pair leakage
measured on the DBpedia set the current paper table was computed from — not a
mistake in `splits_dbpedia.json`, but a property of the data. `build_clean_rq1.py`'s
R3 rule repairs it after the fact; partitioning **connected components** of the
relation makes a crossing pair impossible by construction (1,118 components,
largest 161, no singletons). The builder asserts zero crossing pairs and refuses
to write otherwise.

**Nothing in the shipped benchmark penalises a purely lexical matcher.** 24
distinct negatives across 4,713 groups, none lexically related to their anchors.
`scripts/mine_train_hard_negatives.py` replaces them with same-template,
result-disjoint negatives verified against the oracle (81.5% of same-template
DBpedia pairs are disjoint), giving ~17 per anchor and 1,603 distinct.

> The paper's DBpedia table (MRR@10 0.73 vs 0.67) was measured on the
> 84%-leaked set. `datasets/README.md` already states such numbers "must not be
> used for a new headline result". They are not used here.

---

## 2. Final result

`datasets/dbpedia-v1-leakfree-RQ1.json` — 219 anchors, 588-query pool, 1,515
positives (6.92/anchor), verified leak-free by `check_leakage.py` (0/1,515
memorized pairs). Best configuration for both models, five seeds each.

| DBpedia v1, 5 seeds | MRR@10 | nDCG@10 |
|---|---|---|
| SparqlCE (`ARCH=dbpedia`, 768-d, mined + shared negatives) | 0.6302 ± 0.0363 | 0.6219 ± 0.0333 |
| E5-Finetuned (humanized) | **0.6438** ± 0.0175 | **0.6456** ± 0.0169 |
| seed-level paired *t*-test (one-sided) | p = 0.735 | p = 0.878 |

Both gaps are smaller than SparqlCE's own standard deviation. **Nothing is
significant in either direction.**

For contrast, Wikidata under the older protocol (128-d, no architecture fixes):
SparqlCE 0.6328 ± 0.0631 vs 0.5543 ± 0.1138 MRR@10, also not significant
(p = 0.078–0.117). `experiments-v2.tex` already reads Wikidata as
indistinguishable.

### Ablation ladder (DBpedia v1, nDCG@10, 5 seeds each)

| configuration | SparqlCE | E5-FT |
|---|---|---|
| 128-d, plain | 0.5864 ± 0.024 | 0.6080 |
| 128-d, +query-stream +join-hubs | 0.5806 ± 0.054 | 0.6147 |
| 768-d, +query-stream +join-hubs | 0.6009 ± 0.020 | 0.6098 |
| `ARCH=dbpedia` (adds humanized IRIs, typed edges) | 0.6034 ± 0.036 | 0.6082 |
| + mined hard negatives, own-slice loss | 0.5818 ± 0.043 | 0.6417 ± 0.018 |
| **+ shared negatives** | **0.6219 ± 0.033** | **0.6456 ± 0.017** |

Two effects worth reading off it:

- **Width matters, and the "capacity-matched" claim was wrong.** SparqlCE emitted
  128-d against the control's 768-d. On seed 44 — a *within-seed* contrast, one
  variable changed — nDCG@10 went 0.4949 → 0.6003. Across five seeds the mean
  rose and the standard deviation **halved** (0.054 → 0.020). The bottleneck was
  costing both accuracy and stability.
- **Mined negatives help both, once both can use them.** They lifted the
  baseline +0.034 immediately and SparqlCE only after its loss was allowed to
  score every anchor against every negative in the batch (+0.040).

### Variance

SparqlCE's cross-seed standard deviation is consistently ~2× the control's on
DBpedia (0.033 vs 0.017 here). On Wikidata the published numbers run the other
way, with the control's variance ~3× SparqlCE's. Worth reporting as a
graph-dependent property rather than a constant of either method.

---

## 3. Why DBpedia is flat — the mechanism

Measured over the 4,727 distinct corpus queries, before any model was trained:

| property | DBpedia | Wikidata |
|---|---|---|
| single triple pattern | 1,957 (41%) | — |
| two triple patterns | 2,142 | — |
| three or more | 628 | — |
| `GROUP BY` / `COUNT` / `BIND` / aggregates | **0.0%** | — |
| no join vertex at all | **49.1%** | — |
| graph diameter ≤ 1 (clique joins) | **38.2%** | **4.2%** |
| graph diameter ≤ 1 (join hubs) | 3.6% | 4.0% |

A second GATv2 layer cannot propagate anything across a diameter-1 graph, and
half these queries have no join for it to propagate over. `RUN_GUIDE.md`
independently records that the graph-builder fixes changed nothing on DBpedia
because "every affected construct occurs in 0.0% of the DBpedia corpus" — the
same fact, arrived at twice.

Result-equivalence in this corpus is also mostly **not** a query property:

| pair type | share |
|---|---|
| same template, different constants (**axis C**) | 61–72% |
| genuinely different structure (**axis S**) | 28–39% |

Axis C is decided by whether two constants happen to share an extension in the
graph — knowledge about entities, not about queries. A graph encoder has no
particular advantage there, which is what the flat result reflects.

**The join-hub fix works and is correctly targeted.** Routing each shared
variable through one hub instead of a clique takes DBpedia's diameter-≤1 share
from 38.2% to 3.6% while leaving Wikidata at 4.0%. A fix that only moves the
graph it was predicted to move.

---

## 4. Two attempts to build a corpus that separates them

Both failed, and both failed the way the mechanism predicts.

### 4a. Select structurally rich queries — `scripts/select_structured_queries.py`

Filtering for ≥2 triple patterns, ≥1 join vertex and builder diameter ≥2 keeps
2,285 of 4,727 queries and transforms the surface profile (0% single-pattern, 0%
without a join, OPTIONAL 13%→27%, UNION 10%→18%).

It also **destroys the structural axis**:

| | full corpus | structured subset |
|---|---|---|
| distinct templates | 534 | **39** |
| axis-S share of pairs | 38.6% | **0.0%** |

In these logs, complexity and template diversity are *anti-correlated*: the
multi-pattern queries are a few machine-generated families repeated with
different constants. The structurally different pairs live among the simple
queries the filter removes. Measured by `scripts/check_pair_density.py`.

### 4b. Construct rewrites — `scripts/build_rq1_rewrites.py`

Answer-preserving rewrites are axis-S by definition, with labels independent of
what any entity denotes. Using the RQ2 operators and **verifying every rewrite
against the oracle** rather than trusting the operator:

- 15,626 candidates generated from 4,713 anchors; 20,316 answer sets executed
- **2,572 anchors, 8,023 verified positives, 100% axis-S**
- 2,569 components (largest 7), so the split packs almost perfectly
- test split: 255 anchors, 1,088-query pool, **980 rankable candidates per
  anchor against 10 positives** — 1% relevance density, the best-conditioned
  split built in this work

And then: **zero-shot E5 scores MRR@10 = 1.000 on it.** An untrained text
encoder puts a positive at rank 1 for all 255 anchors, so MRR has no
discriminative power at all. nDCG retains some (0.810@10, 0.775@20), but a
benchmark whose untrained baseline is perfect on the headline metric is not
reportable.

The cause is visible in the operator mix — 64% of positives come from rewrites
that are lexically near-copies:

| operator | verified positives | text change |
|---|---|---|
| `distinct_toggle` | 2,572 | one token |
| `union_duplication` | 2,572 | body repeated verbatim |
| `redundant_optional` | 2,217 | one added clause |
| `triple_shuffle` | 662 | reordering only |
| `subquery_wrapping` | 4 | invalid on 4,709 queries |
| `bgp_split` | 0 | no-op on 4,608 queries |

Genuinely hard rewrites — property path vs explicit join, `UNION` vs
`FILTER(||)`, subquery vs join — need complex source queries, which is exactly
what `subquery_wrapping` and `bgp_split` failing tells us this corpus lacks.

> **Two operators are excluded by design.** `variable_renaming` and
> `filter_true` are normalised away by the graph builder
> (`canonicalize_variables`, `is_noop_filter_text`), so SparqlCE embeds those
> pairs identically *before training* while a text baseline sees different
> strings. Including them would manufacture a win that measures preprocessing.
> They belong in `scripts/verify_invariances.py`, which already reports them.

**These 8,023 verified pairs are a strong RQ2 (robustness) set** — larger than
the current 2,539 and oracle-verified rather than assumed. Measuring embedding
stability under an answer-preserving rewrite is a robustness question; retrieval
rank is the wrong instrument for it.

---

## 5. Measurement bugs found and fixed

Each of these silently distorted results before it was caught.

| bug | effect | fix |
|---|---|---|
| Validation metric credited only the diagonal | Val MRR capped ~0.09 regardless of quality; **both** trainers early-stopped at epoch 1–2 | credit any of the anchor's positives (`e3ea5d4`, `ca152f7`) |
| Validation resampled positives each epoch | same checkpoint scored differently per pass; early stopping reacted to sampling noise | `deterministic=True` eval views |
| Negative parity, component protocol | baseline used curated negatives, SparqlCE did not | `1b9c26f` |
| Negative parity, group protocol | same bug, other branch — baseline +0.034, SparqlCE +0.000 | `713fe79` |
| Hard-negative exposure | baseline's loss scored every anchor against all batch negatives; SparqlCE's only its own slice — 64× at batch 128 | `--shared-negatives` (`67816e9`) |
| Embedding width | 128-d vs the control's 768-d in a "capacity-matched" comparison | `--output-dim` (`cdc2e0c`) |
| Query-stream truncation | fixed 256-token padding truncated >5% of queries — the longest ones — while wasting ~3.5× memory (median is 73 tokens, max 364) | per-batch padding, budget 384 (`d31ec9d`) |
| Text stream read node labels, not the query | SparqlCE was strictly less informed than the baseline it was measured against | `--query-stream` |
| Inference tokenized node labels at 16 vs training's 32 | long labels encoded differently at inference | fixed |
| OOM fallback indexed a `(embedding, gate)` tuple | the recovery path raised instead of recovering | fixed |
| Checkpoint/result path collisions | a variant run silently overwrote another's checkpoint, and a 768-d sweep destroyed the 128-d exports | variant/width/corpus in the path (`f425300`, `a420372`, `3d6d860`, `07dde19`) |
| Split-builder corpus | `--result-sets` alone partitioned the *default* corpus against an unrelated oracle | `--triplets` (`1c2d745`) |

---

## 6. Limitations

- **Not significant, either way.** The DBpedia result is a tie, not a win for
  the baseline. Five seeds cannot resolve a 0.02 difference against a 0.033
  standard deviation.
- **The component protocol's split saturates.** On
  `dbpedia-testsplit-RQ1.json` SparqlCE leads nDCG significantly (p = 0.014–0.043),
  but E5-FT scores exactly **1.000** on that split's axis-S stratum, so it
  cannot separate models. The v1 group split, where both models sit near 0.62,
  is the one that discriminates. Results from the two are not comparable
  (pool 269 vs 588; 12.6 vs 6.9 positives per anchor).
- **The baseline has had no hyperparameter search.** SparqlCE's width was tuned;
  E5-FT runs at the batch-scaled LR and stops at epoch 2. An LR sweep is needed
  before the comparison is beyond question.
- **Wikidata has not been re-run** with the width, query-stream, shared-negative
  or humanization fixes. Its published numbers are 128-d without any of them.
- **`dbpedia_testset_RQ2.json` is stale** — keyed to the old `test_idx`, which no
  longer exists.

---

## 7. Reproducing

```bash
# oracle over the whole corpus (~30-60 min; 4,727 queries, 100% coverage)
python scripts/build_result_sets.py --kg dbpedia \
  --split datasets/dbpedia-triplets-dataset.json \
  --out datasets/result_sets_dbpedia_v2.json --pause 0.1

# leak-free group split (219 anchors) + audit
python scripts/build_clean_rq1.py --kg dbpedia --min-result-size 0 \
  --out datasets/dbpedia-v1-leakfree-RQ1.json
python scripts/check_leakage.py --kg dbpedia \
  --split datasets/dbpedia-v1-leakfree-RQ1.json      # expect CLEAN

# mined hard negatives for the training groups (eval split untouched)
python scripts/mine_train_hard_negatives.py --kg dbpedia \
  --result-sets datasets/result_sets_dbpedia_v2.json \
  --out datasets/dbpedia-triplets-hardneg.json

# the headline table
MAX_NEG=2 DATA=datasets/dbpedia-triplets-hardneg.json ARCH=dbpedia \
  ./scripts/run_seed_sweep.sh dbpedia 5 datasets/dbpedia-v1-leakfree-RQ1.json 3 128 60
```

Diagnostics: `scripts/select_structured_queries.py` (structural profile of any
corpus), `scripts/check_pair_density.py` (pair density and axis split),
`scripts/build_rq1_rewrites.py` (verified answer-preserving rewrites).

---

## 8. What would move DBpedia

Not more tuning — six configurations across five seeds all landed in the same
place.

1. **A structurally rich corpus.** LSQ 2.0 (11.56M queries, 27 endpoints)
   records `joinVertexCount`, `usesFeature`, `resultCount` per query, so a
   subset with real join structure *and* template diversity can be selected
   declaratively. The public endpoint (`lsq.aksw.org/sparql`) is currently
   unreachable; its authors include Saleem and Ngonga Ngomo at DICE.
2. **Wikidata with the four fixes.** It has the constructs DBpedia lacks (4.2%
   diameter ≤1 vs 38.2%) and starts from a +0.079 MRR@10 gap. Halving variance,
   as the width fix did on DBpedia, is what turns p ≈ 0.09 into significance.
3. **A baseline LR sweep**, so the comparison cannot be attributed to an
   undertuned control.
