# RQ1 evaluation splits — `{dbpedia,wikidata}-testsplit-RQ1.json`

The canonical RQ1 retrieval splits, built by
[`scripts/build_clean_rq1.py`](../scripts/build_clean_rq1.py) with **one
identical protocol per knowledge graph** — the same command, changing only
`--kg`, produces both files. They are deterministic: rebuilding reproduces them
byte-for-byte.

```bash
python scripts/build_clean_rq1.py --kg wikidata --min-result-size 0 --out datasets/wikidata-testsplit-RQ1.json
python scripts/build_clean_rq1.py --kg dbpedia  --min-result-size 0 --out datasets/dbpedia-testsplit-RQ1.json
python scripts/check_leakage.py  --kg wikidata --split datasets/wikidata-testsplit-RQ1.json   # expect CLEAN
```

## Composition

| | Wikidata | DBpedia |
|---|---|---|
| anchors | 292 | 219 |
| candidate pool (unique queries) | 448 | 588 |
| positives (total) | 629 | **2,071** (1,515 + 556 relabelled) |
| positives/anchor | mean 2.15, median 1, max 20 | mean 9.46, max 20 |
| unjudged exclusions | 0 (not measurable) | 547 across 50 anchors |
| negatives/anchor | 20 | 20 |
| source | `test_idx` of `splits_{kg}.json` (train never used) | same |
| memorized pairs | **0 / 629** | **0 / 2,071** |

**Report the pool size next to any MRR from these files.** Retrieval scores
depend strongly on how many candidates were ranked; a number from the 448-query
Wikidata pool is not comparable to one from the earlier 1,968-query corpus, nor
to DBpedia's 588.

Lexical spread (anchor→positive TF-IDF cosine on raw text, reported to show no
lexical filter was applied):

| | `<0.3` | `0.3–0.5` | `0.5–0.8` | `≥0.8` |
|---|---|---|---|---|
| Wikidata (mean 0.551) | 4.1% | 48.6% | 25.7% | 21.6% |
| DBpedia (mean 0.540) | 10.5% | 16.0% | 68.9% | 4.6% |

Note the `<0.3` stratum is small on both graphs (12 and 23 anchors). Per-bin
adversarial claims are **not** supportable at that size — see R3 below for why
those anchors mostly disappeared.

## The three construction rules

All fixed a priori; none refers to any model's output.

**R1 — non-singleton results** (`--min-result-size`, **disabled** for these
files). `answer_jaccard > 0.8` between two single-row results only means "the
same one answer", which collapses to identical (J=1) or disjoint (J=0) and never
tests set overlap. It is disabled here because on Wikidata it needs the ~6 GB
execution oracle and, worse, it correlates with high lexical similarity (the
multi-row queries are near-duplicate BIND/path template families), skewing the
split toward the easy `≥0.8` bin.

**R2 — no scalar-aggregate collisions.** Pairs where both queries return a
single aggregate row (COUNT/SUM/AVG/MIN/MAX with no GROUP BY) are dropped:
`answer_jaccard = 1.0` there records that two *numbers* coincided, not that the
queries are result-equivalent. In the raw Wikidata test set 40.2% of
anchor–positive pairs are of this form.

**R3 — no memorized pairs** (`--leak-filter`, on by default). `splits_{kg}.json`
partitions anchor *groups*, but SPARQL query text recurs across groups, so a
held-out group can restate a pair that was optimized over during training. This
is not an unfairness between two models — both train on the same pairs — but it
does not help them equally, and it decides whether a retrieval number is a
retrieval number or a memorization number. Measured on the splits these files
replaced:

| earlier split | eval positive pairs also trained on |
|---|---|
| `wikidata_testset_RQ1_clean.json` | 1,635 / 1,935 (84.5%) |
| `dbpedia_testset_RQ1_clean.json` | 1,116 / 1,325 (84.2%) |

On the 5-seed Wikidata sweep that inflation was worth **+0.32 MRR@10** to
SparqlCE on leaked anchors versus **+0.06** on the rest (E5-Finetuned scored
*lower* on leaked anchors than on clean ones). R3 drops any positive whose
{anchor, positive} pair — whitespace-normalized, direction-agnostic — occurs in
the training groups, and the builder asserts that zero survive.

`--leak-splits` decides whether the validation groups also count as seen. The
default is `train+valid`, because both trainers select their checkpoint on
validation MRR, so a validation pair leaks through model selection.

## R4 — complete judgments over the pool (DBpedia only, so far)

Positives originally came from the triplet file's precomputed lists, never from
the split's actual candidate pool. But the pool is built from *every* anchor's
queries, so anchor A's pool contains anchor B's positives — and some of those are
result-equivalent to A too. Measured with the execution oracle on DBpedia: 1,103
pool queries across 48% of anchors had answer-set Jaccard > 0.8 to their anchor
while not being labelled positive for it. By the benchmark's own criterion those
were correct answers scored as errors.

Incomplete judgments do not penalise systems equally: the model that is *better*
at surfacing genuine result-equivalence is the one ranking unlabelled-but-correct
queries into its top-k, so it is penalised most — precisely the axis RQ1 exists
to measure. [`scripts/relabel_rq1_split.py`](../scripts/relabel_rq1_split.py)
fixes it by evaluating the Jaccard criterion exhaustively over the pool. Of the
1,103:

- **556 became positives** (40% of anchors; positives/anchor 6.92 → 9.46).
- **547 became `excluded_queries`** — result-equivalent *and* memorized training
  pairs. Labelling them reintroduces leakage; leaving them in the ranking scores
  a correct retrieval as an error. They are marked unjudged and dropped from that
  anchor's ranking, the standard treatment of unjudged documents in a pooled IR
  evaluation (`evaluate_rq1.build_excluded_maps`). Effective candidates per
  anchor: 569–587 of 588.

Wikidata is **not** relabelled: its oracle covers only 55% of the pool (2,229
queries total), so 41% of anchors cannot be verified. Fixing that needs a rebuilt
oracle — [`scripts/build_result_sets.py`](../scripts/build_result_sets.py).

## Known properties worth disclosing

- **Negatives are a shared pool.** The 20 negatives per anchor resolve to only
  21 distinct queries across the whole split (both graphs ship them this way).
  The candidate pool is therefore essentially *anchors + their positives*, and
  it shrinks with the anchor count — which is why these splits have smaller
  pools than the sets they replaced.
- **Median 1 positive per anchor on Wikidata**, so for most anchors there is a
  single correct answer in the pool and nDCG tracks MRR closely.
- **Anchors come from `test_idx` only.** `--splits test+valid` is accepted but
  adds nothing once R3 runs with the default `train+valid`, since every
  validation pair is then excluded anyway.

## Audit

[`scripts/check_leakage.py`](../scripts/check_leakage.py) re-measures any split
and, given exported per-anchor scores, breaks each model down by leaked vs
non-leaked anchors. Exit codes: 0 clean, 2 leakage found, 1 audit failed.
`run_seed_sweep.sh` runs it as a pre-flight and refuses a leaky split unless
`ALLOW_LEAKY_SPLIT=1`.
