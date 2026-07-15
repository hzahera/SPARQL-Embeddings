# Evaluation for Research Question 2 (RQ2)

RQ2: how robust are SparqlCE embeddings to structural query perturbations
(variable renaming, triple reordering, redundant `OPTIONAL`, `UNION`
duplication, subquery wrapping, `FILTER(true)`, `DISTINCT` toggling) that
preserve the query's answer set?

## Contents

- **`DBpedia/`** — robustness evaluation and visualization for DBpedia. See
  [DBpedia/README.md](DBpedia/README.md).
- **`Wikidata/`** — robustness evaluation and visualization for Wikidata. See
  [Wikidata/README.md](Wikidata/README.md).

## Data

- `datasets/dbpedia_testset_RQ2.json`, `datasets/wikidata_testset_RQ2.json` —
  perturbation pairs (`original`, `perturbed`, `perturbation_type`), generated
  by `scripts/generate_hard_robustness_dataset.py` /
  `scripts/generate_robustness_wikidata_RQ2.py`.

## Requirements

A trained SparqlCE checkpoint under `sparqlce/models/` (not tracked in git —
see the main [README](../../README.md)).
