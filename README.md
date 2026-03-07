# SPARQLCE — SPARQL Query Embeddings with Contrastive Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

SPARQLCE learns dense vector representations of SPARQL queries grounded in **result-set semantics**. Two queries are considered similar if their answer sets over a knowledge graph overlap (Jaccard similarity). The encoder combines **CodeBERT** token representations with a **GATv2** graph neural network over the SPARQL algebra graph, trained with an **in-batch InfoNCE** contrastive objective.

## Architecture

![SPARQLCE Architecture](docs/architecture.png)

The pipeline proceeds in three stages:

- **A. Data Preprocessing & Triplet Mining** — Collect SPARQL queries from LSQ logs, execute against a SPARQL endpoint, compute pairwise Jaccard similarity over result bindings, and construct (anchor, positive, negative) triplets.
- **B. SPARQL-to-Graph Encoding** — Parse each query into an algebra tree, convert to a directed graph, encode node/edge labels with CodeBERT, and apply GATv2 layers to produce a pooled embedding.
- **C. Contrastive Optimization** — Train with in-batch InfoNCE loss so that queries with overlapping result sets are geometrically close in embedding space.

## Project Structure

```
SPARQL-Embeddings/
├── dataset/
│   ├── dbpedia/
│   │   ├── 01_query_collection.ipynb
│   │   └── 02_triplet_mining.ipynb
│   ├── wikidata/
│   │   ├── 01_query_collection.ipynb
│   │   └── 02_triplet_mining.ipynb
│   ├── raw/
│   │   ├── jaccard_triplets_dbpedia.json
│   │   └── jaccard_triplets_wikidata.json.gz
│   └── splits/
│       ├── splits_dbpedia.json
│       └── splits_wikidata.json
├── src/
│   ├── __init__.py
│   ├── config.py                  # Hyperparameters & paths
│   ├── sparql_graph_builder.py    # SPARQL → NetworkX graph
│   ├── hybrid_encoder.py          # HybridSPARQLEncoder (CodeBERT + GATv2)
│   ├── triplet_dataset.py         # SemanticSPARQLDataset (PyG)
│   ├── embedder.py                # SPARQLEmbedder inference wrapper
│   └── requirements.txt           # Pinned versions (legacy)
├── training/
│   ├── 01_train_dbpedia.ipynb
│   └── 02_train_wikidata.ipynb
├── evaluation/
│   ├── 01_evaluate_dbpedia.ipynb
│   └── 02_evaluate_wikidata.ipynb
├── checkpoints/               # Saved model weights (not tracked in git)
├── docs/
│   └── architecture.png
├── requirements.txt           # Install from here
├── CITATION.cff
├── STRUCTURE.md
└── README.md
```

## Quickstart

### 1. Install

```bash
git clone https://github.com/hzahera/SPARQL-Embeddings.git
cd SPARQL-Embeddings
pip install -r requirements.txt
```

### 2. Run the pipeline

> Run all notebooks from the **project root** so that relative paths resolve correctly.

| Step | Notebook | Output |
|---|---|---|
| 1 | `dataset/dbpedia/01_query_collection.ipynb` | Fetches queries, executes against DBpedia endpoint |
| 2 | `dataset/dbpedia/02_triplet_mining.ipynb` | `dataset/raw/jaccard_triplets_dbpedia.json` |
| 3 | `training/01_train_dbpedia.ipynb` | `checkpoints/triplet_model_semantic-20.pt` + `dataset/splits/splits_dbpedia.json` |
| 4 | `evaluation/01_evaluate_dbpedia.ipynb` | Metric tables in `evaluation/results/` |

### 3. Encode a query

```python
from src.embedder import SPARQLEmbedder

embedder = SPARQLEmbedder("checkpoints/triplet_model_semantic-20.pt")
query = "SELECT ?person WHERE { ?person a <dbo:Scientist> . ?person <dbo:birthPlace> ?place }"
embedding = embedder.encode(query)  # shape: (128,)
print(embedding.shape)
```

## Model

**HybridSPARQLEncoder** (`src/hybrid_encoder.py`)

| Component | Details |
|---|---|
| Text encoder | `microsoft/codebert-base` (768-dim) |
| Graph encoder | GATv2Conv — 2 layers, 4 heads, 256 hidden dim |
| Pooling | Global mean pool over graph nodes |
| Projection | Linear(256×4 → 128), L2-normalized |
| Loss | In-batch InfoNCE (temperature = 0.05) |
| Optimizer | Adam, lr = 1e-5, weight decay = 1e-5 |

**Baselines:** Qwen3-Embedding, MiniLM, E5, MPNet, TF-IDF (unigrams + bigrams).

## Evaluation

All metrics are computed over the held-out test split. SPARQL results are cached; embeddings are computed once per model.

| Metric | Description |
|---|---|
| **Avg Jaccard@k** | Mean result-set Jaccard of top-k retrieved queries |
| **nDCG@k** | Normalized DCG using Jaccard as graded relevance |
| **P / R / F1@k** | Binary precision/recall/F1 (Jaccard ≥ 0.8 = relevant) |

## Dataset

- **Source:** [LSQ — Linked SPARQL Queries](http://lsq.aksw.org/) (DBpedia & Wikidata query logs)
- **Triplet thresholds:** Jaccard ≥ 0.8 → positive; Jaccard ≤ 0.1 → negative
- **Splits:** 70% train / 15% validation / 15% test (stratified by anchor)
- **Raw triplets:** `dataset/raw/`
- **Index splits:** `dataset/splits/`

## Configuration

All hyperparameters and paths are centralised in `src/config.py`:

```python
MODEL_NAME            = "microsoft/codebert-base"
EMBEDDING_DIM         = 768    # CodeBERT output dimension
HIDDEN_DIM            = 256    # GATv2 hidden dimension
OUTPUT_DIM            = 128    # Final embedding dimension
GNN_HEADS             = 4
BATCH_SIZE            = 4
LEARNING_RATE         = 1e-5
EPOCHS                = 20
TEMPERATURE           = 0.05
DATASET_PATH_DBPEDIA  = "dataset/raw/jaccard_triplets_dbpedia.json"
DATASET_PATH_WIKIDATA = "dataset/raw/jaccard_triplets_wikidata.json.gz"
SPLITS_DIR            = "dataset/splits"
CHECKPOINT_DIR        = "checkpoints"
```

## Requirements

- Python 3.10+, PyTorch 2.0+, PyTorch Geometric 2.3+
- `transformers`, `rdflib`, `networkx`, `SPARQLWrapper`
- `sentence-transformers`, `scikit-learn`, `pandas`, `matplotlib`
- CUDA GPU recommended for training

Install all dependencies: `pip install -r requirements.txt`

## Citation

If you use this work, please cite — see [CITATION.cff](CITATION.cff) for the full citation details.

## License

MIT — see [LICENSE](LICENSE).
