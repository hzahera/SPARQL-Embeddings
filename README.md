# SPARQLCE — SPARQL Query Embeddings with Contrastive Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A neural pipeline for learning dense embeddings of SPARQL queries, where similarity is grounded in **result-set overlap** (Jaccard similarity). The model combines **CodeBERT** token representations with a **Graph Attention Network (GAT)** over the SPARQL algebra graph, trained with **in-batch InfoNCE** contrastive loss.

---

## Architecture

![SPARQLCE Architecture](docs/architecture.png)

The pipeline has three stages:
- **A. LSQ Data Preprocessing & Triplet Mining** — Fetch SPARQL queries from LSQ, execute against a SPARQL endpoint, compute pairwise Jaccard similarity on result bindings, and mine (anchor, positive, negative) triplets.
- **B. SPARQL-to-Graph Encoder & Embedding** — Parse each query into an algebra tree, convert to a graph, encode node/edge labels with CodeBERT, and apply GATv2 layers to produce a pooled embedding.
- **C. Contrastive Embedding Space Optimization** — Train with InfoNCE loss so queries with similar result sets are close in embedding space and dissimilar queries are far apart.

---

## Project Structure

```
SPARQL-Embeddings/
├── data/                          # Query collection & triplet mining
│   ├── dbpedia_queries/           # Step 1 & 2 notebooks for DBpedia
│   │   ├── 01_query_collection.ipynb      # Fetch & filter LSQ queries
│   │   └── 02_semantic_triplet_mining.ipynb  # Build Jaccard-based triplets
│   ├── wikidata_queries/          # (Planned) Wikidata equivalent
│   ├── successful_queries/        # JSON: queries with non-empty results
│   └── semantic_triplets/         # JSON: (anchor, positives, negatives)
├── src/                           # Core Python library
│   ├── config.py                  # Hyperparameters & paths
│   ├── sparql_graph_builder.py    # SPARQL → NetworkX graph conversion
│   ├── hybrid_encoder.py          # HybridSPARQLEncoder (CodeBERT + GAT)
│   ├── triplet_dataset.py         # SPARQLDataset (PyG Data objects)
│   ├── embedder.py                # SPARQLEmbedder inference wrapper
│   └── requirements.txt           # Python dependencies
├── training/                      # Training notebooks & data splits
│   ├── train_dbpedia.ipynb        # Train on DBpedia triplets
│   └── splits/                    # Train/valid/test JSON splits
├── evaluation/                    # Evaluation notebooks & results
│   ├── evaluate_dbpedia.ipynb     # Evaluate model vs baselines
│   └── results/                   # Metric tables (CSV + images)
├── checkpoints/                   # Saved model weights (.pt)
├── docs/
│   └── architecture.png           # Pipeline architecture diagram
├── STRUCTURE.md                   # Detailed folder reference
├── LICENSE
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
git clone https://github.com/hzahera/SPARQL-Embeddings.git
cd SPARQL-Embeddings
pip install -r src/requirements.txt
```

### 2. Run the pipeline (in order)

| Step | Notebook | Output |
|------|----------|--------|
| 1 | `data/dbpedia_queries/01_query_collection.ipynb` | `data/successful_queries/successful_queries_dbpedia.json` |
| 2 | `data/dbpedia_queries/02_semantic_triplet_mining.ipynb` | `data/semantic_triplets/semantic_triplets_dbpedia.json` |
| 3 | `training/train_dbpedia.ipynb` | `checkpoints/triplet_model_semantic-20.pt` |
| 4 | `evaluation/evaluate_dbpedia.ipynb` | Metric tables under `evaluation/results/` |

> Run all notebooks from the **project root** so that relative paths resolve correctly.

### 3. Encode a query

```python
from src.embedder import SPARQLEmbedder

embedder = SPARQLEmbedder("checkpoints/triplet_model_semantic-20.pt")

query = "SELECT ?person WHERE { ?person a <dbo:Athlete> . ?person <dbo:birthPlace> <dbr:Cairo> }"
embedding = embedder.encode(query)   # shape: (128,)
print(embedding.shape)
```

---

## Model

**HybridSPARQLEncoder** (`src/hybrid_encoder.py`)

| Component | Details |
|-----------|---------|
| Text encoder | `microsoft/codebert-base` (768-dim, fine-tuned) |
| Graph encoder | GATv2Conv — 2 layers, 4 heads, 256 hidden dim |
| Pooling | Global mean pool over all graph nodes |
| Projection | Linear(256×4 → 128), L2-normalized output |
| Loss | In-batch InfoNCE (temperature = 0.05) |
| Optimizer | Adam, lr = 1e-5, weight decay = 1e-5 |

**Baselines compared:** Qwen3-Embedding, MiniLM, E5, MPNet, TF-IDF (word 1–2 grams).

---

## Evaluation Metrics

All metrics are computed over a pool of unique queries from the test split. SPARQL results are cached; embeddings are computed once per model.

| Metric | Description |
|--------|-------------|
| **avg Jaccard@k** | Mean result-set Jaccard of top-k retrieved queries |
| **nDCG@k** | Normalized DCG using Jaccard as graded relevance |
| **P / R / F1@k** | Binary precision/recall/F1 at k (Jaccard ≥ 0.8 = relevant) |

---

## Dataset

- **Source:** [LSQ (Linked SPARQL Queries)](http://lsq.aksw.org/) — DBpedia query logs
- **Triplet mining:** Jaccard ≥ 0.8 → positive; Jaccard ≤ 0.1 → negative
- **Splits:** ~70% train / 15% valid / 15% test (stratified by anchor)

---

## Configuration

All hyperparameters are in `src/config.py`:

```python
MODEL_NAME     = "microsoft/codebert-base"
EMBEDDING_DIM  = 768    # CodeBERT output dim
HIDDEN_DIM     = 256    # GATv2 hidden dim
OUTPUT_DIM     = 128    # Final embedding dim
GNN_HEADS      = 4
BATCH_SIZE     = 4
LEARNING_RATE  = 1e-5
EPOCHS         = 20
TEMPERATURE   = 0.05
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+, PyTorch Geometric 2.3+
- `transformers`, `rdflib`, `networkx`, `SPARQLWrapper`
- `sentence-transformers`, `scikit-learn`, `pandas`, `matplotlib`
- CUDA GPU recommended for training

See `src/requirements.txt` for pinned versions.

---

## Citation

TBD

---

## License

MIT — see [LICENSE](LICENSE).
