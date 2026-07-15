# Evaluation for Research Question 1 (RQ1)

RQ1: how well do SparqlCE embeddings retrieve queries with overlapping answer
sets, compared to text-only baselines (TF-IDF, BM25, MiniLM, MPNet, E5-base,
Qwen3)?

## Contents

- **`DBpedia/`** — retrieval evaluation and embedding-geometry visualization
  for DBpedia. See [DBpedia/README.md](DBpedia/README.md).
- **`Wikidata/`** — retrieval evaluation and embedding-geometry visualization
  for Wikidata. See [Wikidata/README.md](Wikidata/README.md).
- **`threshold_sensitivity_dbpedia.ipynb`**, **`threshold_sensitivity_wikidata.ipynb`**
  — sensitivity analysis of the SQuARE positive/negative Jaccard thresholds
  (0.8 / 0.2) used to build the training and evaluation labels: the empirical
  Jaccard distribution (bimodality / grey-zone mass) and RQ1 MRR@10 under the
  correctness threshold τ ∈ {0.6, 0.7, 0.8, 0.9}.
- **`evaluate_rq1.py`** — shared evaluation helper (SparqlCE + baseline
  retrieval metrics) imported by the `06_eval_sparqlce_*` notebooks.

## How to run

```bash
pip install -r requirements.txt
cd notebooks/evaluation-RQ1
jupyter lab
```

Requires a trained SparqlCE checkpoint under `sparqlce/models/` (not tracked
in git — see the main [README](../../README.md)) and the datasets under
`datasets/`.
