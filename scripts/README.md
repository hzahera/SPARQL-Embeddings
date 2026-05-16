# Scripts

Utility scripts for dataset generation, evaluation, and analysis.

## Files

- **`extract_unique_queries.py`** — Extract unique SPARQL queries from hard triplet datasets.
  
  Usage:
  ```bash
  python scripts/extract_unique_queries.py --input data/hard_triplets_tfidf_dbpedia.json \
    --output data/unique_queries_dbpedia.json
  ```

- **`robustness_dataset_generation.py`** — Generate perturbation datasets for robustness evaluation (RQ2).

- **`robustness_eval.py`** — Evaluate model robustness on perturbed queries.

- **`generate_hard_robustness_dataset.py`** — Generate hard robustness evaluation datasets.
