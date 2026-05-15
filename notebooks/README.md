# Notebooks

## Structure

- **`evaluation/`** — eval pipelines for TF-IDF, baselines, and SPARQLCE. Why: compute retrieval metrics.
- **`finetuning/`** — binary classifier and contrastive finetuning datasets and training. Why: improve model performance on downstream tasks.
- **`robustness/`** — perturbation datasets and robustness eval. Why: test model robustness to query variations.
- **`training/`** — model training on triplet data. Why: train the core SPARQLCE encoder.
- **`visualization/`** — embedding space visualization. Why: analyze learned representations.

*Note: data preprocessing notebooks have been moved to `data/preprocessing/`.*
