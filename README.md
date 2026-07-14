# SPARQLCE — SPARQL Query Embeddings with Contrastive Learning

SPARQLCE learns dense embeddings for SPARQL queries by combining **CodeBERT text features** with a **GATv2 graph encoder** over query algebra graphs. The model is trained on query pairs whose answer sets overlap, so the embedding space reflects semantic similarity in terms of result-set overlap.

## Updated Project Structure

```
SPARQL-Embeddings/
├── sparqlce/                   # Core Python package
│   ├── config.py               # Hyperparameters and paths
│   ├── graph_builder.py        # SPARQL query → graph conversion
│   ├── encoder.py              # HybridSPARQLEncoder (CodeBERT + GATv2)
│   ├── dataset.py              # SemanticSPARQDataset and collate helpers
│   ├── embedder.py             # SPARQLEmbedder inference wrapper
│   ├── sparql_utils.py         # SPARQL endpoint helpers and caching
│   └── eval_utils.py           # Retrieval metrics and evaluation helpers
├── datasets/                   # Organized datasets
│   ├── binary_datasets/        # Binary classifier datasets
│   ├── dbpedia_query_overlap.json
│   └── sparql2text_dataset.zip
├── notebooks/                  # Research notebooks by task
│   ├── evaluation-RQ1/         # Evaluation for Research Question 1
│   │   ├── DBpedia/            # DBpedia-specific evaluations
│   │   │   └── 06_eval_sparqlce_dbpedia.ipynb
│   │   ├── Wikidata/           # Wikidata-specific evaluations
│   │       └── Multiple evaluation notebooks
│   ├── evaluation-RQ2/         # Evaluation for Research Question 2
│   │   ├── robustness/         # Robustness evaluation
│   │       ├── 01_perturbation_dataset.ipynb
│   │       ├── 02_perturbation_eval.ipynb
│   │       └── evaluate_robustness_v2.py
├── scripts/                    # Supporting scripts for training, evaluation, and analysis (see scripts/README.md)
│   └── training/               # Training-triplet generation
│       └── generate_training_triplets.py
├── docs/                       # Supporting documentation and diagrams
├── requirements.txt            # Runtime dependencies
├── pyproject.toml              # Package metadata
├── CITATION.cff
├── LICENSE
└── README.md
```

## Quickstart

1. Install dependencies:

```bash
git clone https://github.com/hzahera/SPARQL-Embeddings.git
cd SPARQL-Embeddings
pip install -r requirements.txt
```

2. (Optional) Install the package in editable mode:

```bash
pip install -e .
```

3. Make sure the datasets are available under `datasets/`:
- `datasets/dbpedia-triplets-dataset.json`
- `datasets/splits_dbpedia.json`
- `datasets/eval_RQ2_dbpedia.json`
- `datasets/binary_datasets/`

4. Start Jupyter from the project root:

```bash
jupyter lab
```

## Using the library

To use the SPARQLCE model for encoding SPARQL queries into dense embeddings, follow the example below:

```python
from sparqlce.embedder import SPARQLEmbedder

# Initialize the embedder with the path to the trained model checkpoint
embedder = SPARQLEmbedder("checkpoints/<model_checkpoint>.pt")

# Define a SPARQL query
query = "SELECT ?person WHERE { ?person a dbo:Scientist . ?person dbo:birthPlace ?place }"

# Encode the query into a dense embedding
embedding = embedder.encode(query)

# Print the shape of the resulting embedding
print(embedding.shape)
```

### Notes:
- Ensure that the checkpoint path points to a trained model file under `checkpoints/` or elsewhere on disk.
- The `SPARQLEmbedder` class handles the conversion of SPARQL queries into graph representations and uses the trained `HybridSPARQLEncoder` for encoding.

## Core modules

- `sparqlce/config.py` — consolidated hyperparameters, paths, and device selection
- `sparqlce/graph_builder.py` — builds labelled NetworkX graphs from SPARQL queries
- `sparqlce/encoder.py` — hybrid CodeBERT + GATv2 query encoder
- `sparqlce/dataset.py` — triplet dataset loader and PyG collate helpers
- `sparqlce/embedder.py` — inference wrapper for trained checkpoints
- `sparqlce/sparql_utils.py` — SPARQL endpoint utilities and caching
- `sparqlce/eval_utils.py` — retrieval metrics and baseline evaluation helpers

## Scripts

See `scripts/README.md` for the full list (training, evaluation, significance
testing, ablations, and plotting). Highlights:

- `scripts/train_sparqlce.py` — CLI training script (`--kg`, `--seed`, `--ablation`)
- `scripts/training/generate_training_triplets.py` — generate training triplets
- `scripts/verify_invariances.py` — verify model invariances with a checkpoint

## Evaluation

Evaluation workflows are primarily contained in notebooks under `notebooks/`:
- `notebooks/evaluation-RQ1/` — retrieval evaluation notebooks for DBpedia and Wikidata
- `notebooks/evaluation-RQ2/robustness/` — robustness evaluation notebooks

These notebooks are the canonical evaluation artifacts; `scripts/evaluation/` contains a small optional support script only.

## Data overview

- `datasets/dbpedia-triplets-dataset.json` — DBpedia triplet dataset
- `datasets/splits_dbpedia.json` — train/val/test splits
- `datasets/eval_RQ2_dbpedia.json` — robustness evaluation queries
- `datasets/binary_datasets/` — binary classifier dataset splits
- `datasets/result_sets_dbpedia.json` — DBpedia result set metadata

The repository stores datasets under `datasets/`; no `data/` directory is required.

## Notes for research

- Use `sparqlce/` as the importable package for notebooks and scripts.
- `checkpoints/`, `cache/`, and `outputs/` are generated runtime directories and are not committed.
- The repository is organised for reproducibility and clean experimentation.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- `transformers`, `rdflib`, `networkx`, `SPARQLWrapper`
- `sentence-transformers`, `scikit-learn`, `pandas`, `matplotlib`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## License

MIT — see `LICENSE`.

## Citation

If you use this work in your research, please cite it. See `CITATION.cff` for details.

## DBpedia Training and Evaluation

### Training
The training process for DBpedia is implemented in the notebook `notebooks/training/train_dbpedia.ipynb`. The training pipeline includes:
- **Loss Function**: Combines InfoNCE loss with a Jaccard calibration term.
- **Optimizer**: AdamW with differential learning rates for CodeBERT and GNN layers.
- **Dataset**: Utilizes `datasets/dbpedia-triplets-dataset.json` and `datasets/splits_dbpedia.json` for fixed splits.
- **Hyperparameters**: Batch size of 32, temperature of 0.05, and 20 epochs.

### Evaluation
Evaluation for DBpedia is conducted in the following notebooks:
- `notebooks/evaluation-RQ1/DBpedia/06_eval_sparqlce_dbpedia.ipynb`: Evaluates retrieval metrics such as MRR, Hits@1, and Hits@5.
- `notebooks/evaluation-RQ2/robustness/04_eval_sparql_RQ2.ipynb`: Assesses robustness to query perturbations using `datasets/eval_RQ2_dbpedia.json`.

### Results
- **Validation Metrics**: Best validation MRR is used for model selection.
- **Test Metrics**: Final evaluation on the test split reports MRR, Hits@1, and Hits@5.

### Scripts
See `scripts/README.md` for the full list.

- `scripts/training/generate_training_triplets.py`: Generates triplets for training.
