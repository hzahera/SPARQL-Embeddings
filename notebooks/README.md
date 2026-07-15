# Notebooks

- **`training/`** — model training on triplet data (DBpedia, Wikidata). See
  [training/README.md](training/README.md).
- **`finetuning/`** — contrastive fine-tuning of the Qwen3 baseline on SQuARE.
  See [finetuning/README.md](finetuning/README.md).
- **`evaluation-RQ1/`** — RQ1 retrieval evaluation and embedding-geometry
  visualization, DBpedia and Wikidata. See
  [evaluation-RQ1/README.md](evaluation-RQ1/README.md).
- **`evaluation-RQ2/`** — RQ2 robustness-to-perturbation evaluation, DBpedia
  and Wikidata. See [evaluation-RQ2/README.md](evaluation-RQ2/README.md).
- **`caching/`** — downstream semantic query-caching simulation. See
  [caching/README.md](caching/README.md).
- **`visualization/`** — embedding space visualization. See
  [visualization/README.md](visualization/README.md).

All notebooks expect to be run from within their own directory (or from the
repo root with `PROJECT_ROOT` resolved relative to the notebook) with the
package installed (`pip install -r requirements.txt` from the repo root) and
the relevant datasets present under `datasets/`.
