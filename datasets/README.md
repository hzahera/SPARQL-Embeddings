# Datasets

SQuARE (result-set-grounded SPARQL query pairs) training and evaluation data
for DBpedia and Wikidata.

## Tracked in this repo

- **`dbpedia-triplets-dataset.json`** — DBpedia SQuARE training triplets.
- **`splits_dbpedia.json`**, **`splits_wikidata.json`** — fixed
  train/val/test indices (seed 42).
- **`dbpedia_testset_RQ1.json`**, **`wikidata_testset_RQ1.json`** — RQ1
  retrieval evaluation anchors.
- **`dbpedia_testset_RQ2.json`**, **`wikidata_testset_RQ2.json`** — RQ2
  perturbation pairs (`original`, `perturbed`, `perturbation_type`).
- **`result_sets_dbpedia.json`** — cached true answer sets for the DBpedia
  RQ1 test corpus (used as an execution oracle by the evaluation and caching
  notebooks, so no live SPARQL endpoint is needed).

## Not tracked (exceed GitHub's size limits)

Place these in `datasets/` before running the Wikidata notebooks:

- **`wikidata-triplets-dataset.json`** (~108 MB) — the SQuARE-Wikidata
  training triplets, indexed by `splits_wikidata.json`.
- **`result_sets_wikidata.json`** (~5.9 GB) — cached answer sets for the
  Wikidata RQ1 test corpus. Regenerable by running
  `notebooks/evaluation-RQ1/Wikidata/07_visualize_alignment_wikidata.ipynb`,
  which executes the queries against a Wikidata SPARQL endpoint and writes
  this cache (~45 min). A freshly generated cache reflects the current state
  of that endpoint's dump.

Both files are available from the authors on request.

## Evaluation test sets — construction (finalized, leak-free)

All four evaluation sets are built from the **held-out** `test_idx`
(leak-free: 0 anchors overlap train/val), with a **single, consistent
construction per RQ across both KGs**.

### RQ1 — retrieval (`{dbpedia,wikidata}_testset_RQ1.json`)
- **Anchors:** held-out `test_idx`.
- **Positives (relevant):** answer-set Jaccard > 0.8 (answer-equivalent),
  **no lexical filter** — natural lexical distribution.
- **Negatives (distractors):** the triplets' hard negatives (lexically
  similar / answer different).
- **Reporting:** stratify results by anchor–positive lexical similarity
  (TF-IDF cosine bins) — surface methods win on high-similarity positives and
  collapse on divergent ones; SparqlCE stays flat (see `evaluate_rq1.py`
  `STRATIFY_BINS`).
- Sizes: **DBpedia 462 anchors / 1841 corpus**, **Wikidata 1063 / 3837**.

### RQ2 — robustness (`{dbpedia,wikidata}_testset_RQ2.json`)
- **Originals:** held-out `test_idx` **anchors** only (this is the only
  leak-free choice — the RQ1 corpus includes positives/negatives that are
  often training anchors, so perturbing the full corpus reintroduces leakage).
- **Perturbations:** answer-preserving rewrites — 6 shared operators
  (`variable_renaming`, `triple_shuffle`, `union_duplication`,
  `redundant_optional`, `filter_true`, `distinct_toggle`) + 1 KG-specific
  (`subquery_wrapping` for DBpedia, `bgp_split` for Wikidata).
- Pairs: **DBpedia 2539 (472 originals)**, **Wikidata 4705 (1074 originals)**;
  both **0 train leak**.
- **Note:** DBpedia's `subquery_wrapping` yields only ~1 valid pair — the
  held-out DBpedia anchors are structurally simple (mostly `rdf:type`) and do
  not support it. Report DBpedia RQ2 on the 6 well-covered operators. (The
  paper's earlier `subquery_wrapping` coverage came from perturbing a *leaky*
  corpus, so it is not usable here.)

### RQ1 "hard adversarial" sets (`{dbpedia,wikidata}_testset_RQ1_hard.json`)

A stricter, motivating-example version of RQ1: for each held-out anchor, pick
the single **hardest positive** (answer-Jaccard ≥ 0.8, lexical cosine ≤ 0.3 —
result-equivalent but surface-divergent) and the single **hardest negative**
(lexical cosine ≥ 0.8, answer-Jaccard ≤ 0.2 — surface-similar but
result-different). This is the sharpest statement of the paper's thesis, but
be aware of what the real data supports before quoting it:

- **DBpedia** (`scripts/build_hard_adversarial_rq1.py`, run in-repo,
  verified against the paper's own stored Jaccard values byte-for-byte):
  472 unique held-out anchors is the hard ceiling (that's the entire test
  set). Only **87/472 anchors** actually clear *both* thresholds
  simultaneously. The shipped file has 500 rows because the 472 unique-anchor
  rows are padded with each anchor's 2nd-best (positive, negative) pair,
  ranked by a `hardness_score`, to reach a round number — every row carries
  `meets_strict_bar: true/false` and a `slot` index (0 = anchor's best pair)
  so this is fully auditable. **Report the 87 as the headline "hard
  adversarial" number**; treat the extended 500 as a ranked robustness check,
  not as "500 hard cases."
- **Wikidata** (`scripts/build_hard_adversarial_rq1_wikidata.py`, run on the
  cluster — its cached result sets run up to 6.6M rows per query, so an
  exact corpus-wide Jaccard scan like DBpedia's is not tractable in a
  memory-constrained sandbox): a cheap TF-IDF lexical candidate search over
  the full corpus identified 7,506 distinct queries worth resolving, but
  `result_sets_wikidata.json` only has **2,229 cached queries total**, so
  only 2,229/7,506 (~30%) could be checked with exact Jaccard — the rest
  fell back to weaker candidates. Result: **19/1074 anchors** clear both
  thresholds simultaneously (vs. DBpedia's 87/472) — a real, expected
  consequence of Wikidata's sparser answer-set cache, not a bug. All 500
  shipped rows are distinct anchors (no padding was needed — every one of
  them at least had a fallback candidate); filter to `meets_strict_bar: true`
  for the 19 that meet the actual bar. **Report DBpedia (87) and Wikidata
  (19) side by side** rather than treating them as comparable-scale sets.
