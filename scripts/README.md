# Scripts

Utility scripts for dataset generation, evaluation, training, and analysis.

## Dataset generation

- **`extract_unique_queries.py`** — extract unique SPARQL queries from hard triplet datasets.

  ```bash
  python scripts/extract_unique_queries.py --input data/hard_triplets_tfidf_dbpedia.json \
    --output data/unique_queries_dbpedia.json
  ```

- **`generate_hard_robustness_dataset.py`** — generate hard robustness evaluation datasets.
- **`generate_robustness_wikidata_RQ2.py`** — generate the Wikidata RQ2 perturbation dataset.
- **`add_missing_perturbations.py`** — backfill missing perturbation types into an existing RQ2 dataset.
- **`robustness/robustness_dataset_generation.py`** — provenance script for the RQ2 perturbation datasets.
- **`training/generate_training_triplets.py`** — generate training triplets from unique queries via Jaccard similarity.

## Training

- **`train_sparqlce.py`** — CLI training script (`--kg {dbpedia,wikidata} --seed N [--ablation {full,no_gnn,no_decomposition}]`),
  consolidating `notebooks/training/{train_dbpedia,train_wikidata}.ipynb` into something loopable for seed/ablation sweeps.
- **`run_seed_sweep.sh`** — trains N seeds per KG and aggregates significance stats (Fatal #1 fix; see `docs/significance_and_ablation_plan.md`).
- **`run_ablation_sweep.sh`** — trains the `no_gnn`/`no_decomposition` ablation variants and builds the ablation table (Fatal #2 fix).

## Evaluation

- **`verify_invariances.py`** — checks the graph builder's variable-renaming and `FILTER(true)` invariances.
- **`measure_fallback_fraction.py`** — measures how often the graph builder falls back to a flat token-chain representation.
- **`eval_rq1_export.py`** — RQ1 evaluation with per-anchor score export (needed for paired significance testing; see `notebooks/evaluation-RQ1/evaluate_rq1.py` for the aggregate-only version).
- **`compute_significance.py`** — cross-seed mean/std and a paired Wilcoxon signed-rank test against a baseline.
- **`build_ablation_table.py`** — aggregates per-seed ablation exports into a LaTeX table.

## Plotting

- **`plot_robustness_radar.py`** — per-operator robustness radar (paper appendix figure).
- **`plot_rank_stability.py`** — MRR-before/after-perturbation bar chart.
