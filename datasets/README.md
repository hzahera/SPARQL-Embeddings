# Datasets

This folder contains datasets required for the evaluation of DBpedia and Wikidata.

## Contents
- **dbpedia-triplets-dataset.json**: Contains triplets for DBpedia evaluation.
- **eval_RQ2_dbpedia.json**: Evaluation data for Research Question 2 (DBpedia).
- **result_sets_dbpedia.json**: Result sets for DBpedia evaluation.
- **splits_dbpedia.json**: Data splits for DBpedia evaluation.
- **binary_datasets/**: Contains binary datasets for training and validation.

## Wikidata Datasets
For Wikidata evaluation, ensure the following datasets are available:
- `wikidata-triplets-dataset.json`
- `eval_RQ2_wikidata.json`
- `result_sets_wikidata.json`
- `splits_wikidata.json`

### Large files not tracked in git
Two Wikidata files exceed GitHub's size limits and are therefore
gitignored — obtain them separately and place them in this folder:

- `wikidata-triplets-dataset.json` (~108 MB): the SQuARE-Wikidata
  training triplets (10,735 anchor groups indexed by
  `splits_wikidata.json`). Available from the authors.
- `result_sets_wikidata.json` (~5.9 GB): cached answer sets for all
  2,229 RQ1 test-corpus queries. Regenerable by running
  `notebooks/evaluation-RQ1/Wikidata/07_visualize_alignment_wikidata.ipynb`,
  which executes the queries against
  `https://wikidata.data.dice-research.org/sparql` (~45 min) and writes
  this cache. Note that a freshly generated cache reflects the current
  state of that endpoint's dump.

## Usage
Place the datasets in the appropriate folders before running the evaluation notebooks. Ensure the paths in the notebooks are updated if the datasets are stored in a different location.