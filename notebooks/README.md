# Notebooks

This folder collects Jupyter notebooks used for dataset preparation, training, evaluation, robustness experiments, and visualization.

Structure

- `notebooks/data/` — query collection and triplet mining notebooks.
- `notebooks/finetuning/` — finetuning dataset creation and training notebooks for binary/fine-tuned models.
- `notebooks/evaluation/` — evaluation pipelines for TF-IDF, baselines, and SPARQLCE.
- `notebooks/robustness/` — perturbation dataset generation and robustness evaluation notebooks.
- `notebooks/training/` — model training notebooks.
- `notebooks/visualization/` — visualization and embedding-space analysis notebooks.
- `notebooks/baseline/` — baseline evaluation notebooks.

Quickstart (interactive)

1. Create and activate a Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start Jupyter Lab from the project root:

```bash
jupyter lab
```

3. Open the desired notebook under `notebooks/` and run cells interactively.

Run notebooks non-interactively (headless)

Use `nbconvert` to execute a notebook end-to-end. Example (increase timeout for long runs):

```bash
jupyter nbconvert --to notebook --execute notebooks/evaluation/06_eval_sparqlce_dbpedia.ipynb \
  --ExecutePreprocessor.timeout=600 --output notebooks/evaluation/06_eval_sparqlce_dbpedia.executed.ipynb
```

Notes and recommendations

- Use Python 3.10+ and install dependencies from `requirements.txt`.
- Notebooks assume the package is importable as `sparqlce` (see `pyproject.toml`). If working in editable mode:

```bash
pip install -e .
```

- Large datasets and checkpoints are stored under `data/`, `checkpoints/`, and `cache/` (these are not committed). Outputs and figures are written to `outputs/`.

Suggested order to reproduce core experiments

1. `notebooks/data/01_query_collection_*` — build query collections.
2. `notebooks/data/02_triplet_mining_*` — mine triplets and produce `data/` artifacts.
3. `notebooks/finetuning/*` or `notebooks/training/*` — train models or finetune.
4. `notebooks/evaluation/*` — run evaluation notebooks to compute retrieval metrics and plots.

If you want, I can also add a small example notebook that runs a minimal encode+retrieve smoke test locally.
