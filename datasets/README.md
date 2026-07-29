# Datasets

SQuARE (result-set-grounded SPARQL query pairs) training and evaluation data
for DBpedia and Wikidata. The folder holds only the **final** training and
test splits for RQ1 (retrieval) and RQ2 (robustness).

## Tracked in this repo

- **`dbpedia-triplets-dataset.json`** — DBpedia SQuARE training triplets.
- **`splits_dbpedia.json`**, **`splits_wikidata.json`** — fixed
  train/val/test indices (seed 42).
- **`{dbpedia,wikidata}-testsplit-RQ1.json`** — the canonical RQ1 evaluation
  splits (219 / 292 anchors, 588 / 448-query pools, 0 memorized pairs). See
  [RQ1_testsplits.README.md](RQ1_testsplits.README.md). Use these.
- **`dbpedia_testset_RQ1_LEAKY.json`**, **`wikidata_testset_RQ1_LEAKY.json`** —
  the superseded RQ1 sets, kept ONLY to reproduce numbers already published from
  them (the paper's DBpedia RQ1 table and the Wikidata zero-shot rows). Renamed
  with the `_LEAKY` tag because 59.9% (DBpedia) and 54.6% (Wikidata) of their
  anchor--positive pairs also occur as training pairs, affecting 83.8% and 79.9%
  of anchors: a group-level train/test split does not stop query *text* from
  recurring across groups. Numbers measured on them partly reflect memorization
  and must not be used for a new headline result. Delete once the tables that
  cite them are regenerated (`scripts/rerun_rq1_table.sh`). Old description: RQ1
  retrieval evaluation sets (identical construction across both KGs, see below).
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
  Wikidata RQ1 corpus. Regenerable via
  `notebooks/evaluation-RQ1/Wikidata/07_visualize_alignment_wikidata.ipynb`.

Both files are available from the authors on request.

## Evaluation test sets — construction (finalized, leak-free)

All sets are built from the **held-out** `test_idx` (0 anchors overlap
train/val), with a **single, consistent construction per RQ across both KGs**.

### RQ1 — retrieval (`{dbpedia,wikidata}_testset_RQ1.json`)

Both KGs use the identical protocol so results are directly comparable:

- **Anchors:** 462 held-out `test_idx` anchors per KG.
- **Positives (relevant):** answer-set Jaccard > 0.8 (answer-equivalent), **no
  lexical filter** (natural lexical distribution), capped at 20 per anchor.
- **Negatives (distractors):** 20 per anchor, answer-set Jaccard = 0.
- **Positives per anchor:** 1–20 (DBpedia mean 8.18 / Wikidata mean 7.94),
  matched histogram (1 / 2–4 / 5–9 / 10–19 / 20).
- **Corpus:** DBpedia 1,841 unique queries / Wikidata 1,968.
- **Reporting:** whole-file MRR / Hits / nDCG at @3/@5/@10 (see
  `tab-RQ1-{DBpedia,Wikidata}.tex`).

The Wikidata RQ1 set was constructed to mirror the DBpedia set on size,
positives/negatives, and thresholds by
[`scripts/build_dbpedia_matched_wikidata.py`](../scripts/build_dbpedia_matched_wikidata.py)
(deterministic, seed 42; sampled from `test_idx` only — verified 0 train/val
leak). Evaluate any RQ1 file with
[`scripts/eval_rq1_stratified.py`](../scripts/eval_rq1_stratified.py)
(`--split datasets/<file>.json`). All *text* baselines receive the same
Wikidata label-expanded query text SparqlCE resolves internally
(`expand_wikidata_labels`), for a fair comparison. On this natural-distribution
Wikidata split the fine-tuned E5 baseline leads (MRR@10 0.591) with SparqlCE
second (0.548, tying on Hits@10 at 0.747); see `tab-RQ1-Wikidata.tex`.

### RQ2 — robustness (`{dbpedia,wikidata}_testset_RQ2.json`)

- **Originals:** held-out `test_idx` **anchors** only (the only leak-free
  choice — the RQ1 corpus includes positives/negatives that are often training
  anchors, so perturbing the full corpus reintroduces leakage).
- **Perturbations:** answer-preserving rewrites — 6 shared operators
  (`variable_renaming`, `triple_shuffle`, `union_duplication`,
  `redundant_optional`, `filter_true`, `distinct_toggle`) + 1 KG-specific
  (`subquery_wrapping` for DBpedia, `bgp_split` for Wikidata).
- Pairs: **DBpedia 2539 (472 originals)**, **Wikidata 4705 (1074 originals)**;
  both **0 train leak**.
- **Note:** DBpedia's `subquery_wrapping` yields only ~1 valid pair — the
  held-out DBpedia anchors are structurally simple (mostly `rdf:type`) and do
  not support it. Report DBpedia RQ2 on the 6 well-covered operators.
