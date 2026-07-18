# RQ2 Evaluation — Robustness (DBpedia)

Notebooks for RQ2 ("How robust are SparqlCE embeddings to structural query
perturbations?") on DBpedia.

## Contents

- `04_eval_sparql_RQ2_dbpedia.ipynb` — evaluates SparqlCE and text baselines
  (TF-IDF, MiniLM, MPNet, E5-base, Qwen3) on `datasets/dbpedia_testset_RQ2.json`;
  reports mean cosine similarity and MRR before/after perturbation. Writes
  `results/robustness/robustness_results.json`.
- `04b_eval_qwen3_finetuned_RQ2_dbpedia.ipynb` — same protocol for the
  fine-tuned Qwen3 baseline (LoRA adapter trained by
  `notebooks/finetuning/04_qwen3_contrastive_finetune_square.ipynb`). Writes
  `results/robustness/robustness_qwen3_finetuned_dbpedia.json`.
- `08_visualize_robustness_dbpedia.ipynb` — builds the two-panel robustness
  figure (MRR before/after per model; cosine similarity per perturbation
  type) from `results/robustness/robustness_results.json`.

## How to run

```bash
pip install -r requirements.txt
cd notebooks/evaluation-RQ2/DBpedia
jupyter lab
```

Run `04_eval_sparql_RQ2_dbpedia.ipynb` (and optionally `04b_...`) before
`08_visualize_robustness_dbpedia.ipynb`, since the visualization notebook
reads their output JSON. Requires a trained SparqlCE checkpoint under
`sparqlce/models/` (not tracked in git).
