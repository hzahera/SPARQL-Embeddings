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
- **`binary_datasets/`** — binary classifier train/val splits
  (`dbpedia_train.json`, `dbpedia_val.json`).

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

## RQ1 test sets — construction (SQuARE)

Built from the **held-out** `test_idx` (leak-free: 0 anchors overlap train/val).

- **Wikidata** (`wikidata_testset_RQ1.json`): positives = answer-set Jaccard > 0.8
  (answer-equivalent), **no lexical filter** — natural lexical distribution
  (TF-IDF cosine to anchor: median 0.25, ~70% of positives < 0.4). Hard
  negatives (lexically similar / answer different) kept as distractors.
  1063 anchors / 3837 corpus. Report results stratified by anchor-positive
  lexical similarity to show where surface methods fail and SparqlCE holds.
- **DBpedia** (`dbpedia_testset_RQ1.json`): currently the strict adversarial
  variant (answer > 0.8 AND TF-IDF cosine < 0.3), 123 anchors / 550 corpus.
  TODO: rebuild answer-only to match Wikidata's construction for consistency.
