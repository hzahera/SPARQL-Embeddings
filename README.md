# SPARQLCE — SPARQL Query Embeddings with Contrastive Learning

SPARQLCE learns dense embeddings for SPARQL queries by combining **CodeBERT
text features** with a **GATv2 graph encoder** over query algebra graphs. The
model is trained on query pairs whose answer sets overlap (the SQuARE
supervision signal), so the embedding space reflects semantic similarity in
terms of result-set overlap rather than surface syntax.

![Architecture](docs/architecture.png)

## Project structure

```text
SPARQL-Embeddings/
├── sparqlce/                   # Core Python package
│   ├── config.py               # Hyperparameters and paths
│   ├── graph_builder.py        # SPARQL query -> graph conversion
│   ├── encoder.py               # HybridSPARQLEncoder (CodeBERT + GATv2)
│   ├── dataset.py               # SemanticSPARQDataset and collate helpers
│   ├── embedder.py              # SPARQLEmbedder inference wrapper
│   ├── sparql_utils.py          # SPARQL endpoint helpers and caching
│   └── eval_utils.py            # Retrieval metrics and evaluation helpers
├── datasets/                   # Datasets for DBpedia and Wikidata (see datasets/README.md)
├── notebooks/                  # Research notebooks by task (see notebooks/README.md)
│   ├── training/                # Model training
│   ├── finetuning/              # Qwen3 baseline fine-tuning
│   ├── evaluation-RQ1/          # RQ1: retrieval evaluation
│   ├── evaluation-RQ2/          # RQ2: robustness evaluation
│   ├── caching/                 # Downstream semantic caching simulation
│   └── visualization/           # Embedding space visualization
├── scripts/                    # CLI scripts for training, evaluation, and analysis (see scripts/README.md)
├── results/                     # Evaluation outputs (figures, significance tables, robustness JSON)
├── docs/                        # Architecture diagram
├── requirements.txt             # Runtime dependencies
├── pyproject.toml               # Package metadata
├── CITATION.cff
├── LICENSE
└── README.md
```

## Quickstart

1. Install dependencies:

```bash
git clone <this-repository-url>
cd SPARQL-Embeddings
pip install -r requirements.txt
```

1. (Optional) Install the package in editable mode:

```bash
pip install -e .
```

1. Make sure the datasets are available under `datasets/` — see
   [datasets/README.md](datasets/README.md) for what's tracked in git versus
   what must be obtained separately (two Wikidata files exceed GitHub's size
   limits).

1. Start Jupyter from the project root:

```bash
jupyter lab
```

## Using the library

```python
from sparqlce.embedder import SPARQLEmbedder

# Initialize the embedder with the path to a trained model checkpoint
embedder = SPARQLEmbedder("sparqlce/models/<checkpoint>.pt")

# Define a SPARQL query
query = "SELECT ?person WHERE { ?person a dbo:Scientist . ?person dbo:birthPlace ?place }"

# Encode the query into a dense embedding
embedding = embedder.encode(query)
print(embedding.shape)
```

`SPARQLEmbedder` converts the SPARQL query into a graph representation and
uses the trained `HybridSPARQLEncoder` to produce an L2-normalized embedding.
Trained checkpoints are not tracked in git (see "Trained checkpoints" below).

## Core modules

- `sparqlce/config.py` — consolidated hyperparameters, paths, and device selection
- `sparqlce/graph_builder.py` — builds labelled NetworkX graphs from SPARQL queries
- `sparqlce/encoder.py` — hybrid CodeBERT + GATv2 query encoder (`ablation={full,no_gnn,no_decomposition}`)
- `sparqlce/dataset.py` — triplet dataset loader and PyG collate helpers
- `sparqlce/embedder.py` — inference wrapper for trained checkpoints
- `sparqlce/sparql_utils.py` — SPARQL endpoint utilities and caching
- `sparqlce/eval_utils.py` — retrieval metrics and baseline evaluation helpers

## Training

Training for both knowledge graphs is implemented in
`notebooks/training/{train_dbpedia,train_wikidata}.ipynb`:

- **Loss**: InfoNCE (in-batch contrastive) + 0.3 × Jaccard-calibration MSE term.
- **Optimizer**: AdamW with differential learning rates for CodeBERT and GNN layers.
- **Data**: `datasets/{dbpedia,wikidata}-triplets-dataset.json` with fixed splits from `datasets/splits_{dbpedia,wikidata}.json`.
- **Hyperparameters**: batch size 32, temperature 0.05, 20 epochs.
- **Model selection**: best validation MRR.

`scripts/train_sparqlce.py` provides an equivalent CLI (`--kg {dbpedia,wikidata} --seed N [--ablation ...]`) for seed and ablation sweeps — see `scripts/README.md`.

## Evaluation

Evaluation workflows are primarily notebooks under `notebooks/`:

- `notebooks/evaluation-RQ1/` — retrieval evaluation (MRR, nDCG) and embedding-geometry visualization for DBpedia and Wikidata.
- `notebooks/evaluation-RQ2/` — robustness to structural query perturbations for DBpedia and Wikidata.
- `notebooks/caching/` — downstream semantic query-caching simulation.

`results/significance/` documents multi-seed variance and significance
testing against the strongest baseline; see
[results/significance/README.md](results/significance/README.md).

## Trained checkpoints

Trained model checkpoints (`sparqlce/models/*.pt`) are not tracked in git
(see `.gitignore`) — obtain them from the authors, or produce your own by
running the training notebooks/script above.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- `transformers`, `rdflib`, `networkx`, `SPARQLWrapper`
- `sentence-transformers`, `rank_bm25`, `scikit-learn`, `pandas`, `matplotlib`

```bash
pip install -r requirements.txt
```

## License

MIT — see `LICENSE`.

## Citation

If you use this work in your research, please cite it. See `CITATION.cff` for
details.
