# RQ2 Evaluation — Robustness (Wikidata)

Notebooks for RQ2 ("How robust are SparqlCE embeddings to structural query
perturbations?") on Wikidata.

## Contents

- `04_eval_sparql_RQ2_wikidata.ipynb` — evaluates SparqlCE and text baselines
  (TF-IDF, MiniLM, MPNet, E5-base, Qwen3) on `datasets/wikidata_testset_RQ2.json`;
  reports mean cosine similarity and MRR before/after perturbation. Writes
  `results/robustness/robustness_results_wikidata.json`.
- `09_eval_qwen3_finetuned_RQ2_wikidata.ipynb` — same protocol for the
  fine-tuned Qwen3 baseline (LoRA adapter trained by
  `notebooks/finetuning/05_qwen3_contrastive_finetune_square_wikidata.ipynb`).
  Writes `results/robustness/robustness_qwen3_finetuned_wikidata.json`.
- `08_visualize_robustness_wikidata.ipynb` — builds the two-panel robustness
  figure from `results/robustness/robustness_results_wikidata.json`.

## How to run

```bash
pip install -r requirements.txt
cd notebooks/evaluation-RQ2/Wikidata
jupyter lab
```

Run `04_eval_sparql_RQ2_wikidata.ipynb` (and optionally `09_...`) before
`08_visualize_robustness_wikidata.ipynb`, since the visualization notebook
reads their output JSON. Requires a trained SparqlCE checkpoint under
`sparqlce/models/` (not tracked in git).
