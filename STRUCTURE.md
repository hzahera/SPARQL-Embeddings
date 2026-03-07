# SPARQLCE — Folder & File Reference

## Top-level layout

```
SPARQL-Embeddings/
├── data/                    # Query collection & triplet mining
├── src/                     # Core Python library
├── training/                # Training notebooks & data splits
├── evaluation/              # Evaluation notebooks & results
├── checkpoints/             # Saved model weights
├── docs/                    # Documentation assets (architecture diagram)
├── README.md
├── STRUCTURE.md             # This file
└── LICENSE
```

---

## data/

Holds everything related to query collection and triplet mining.

| Path | Description |
|------|-------------|
| `data/dbpedia_queries/01_query_collection.ipynb` | Fetch SPARQL queries from LSQ, execute against DBpedia, save valid ones |
| `data/dbpedia_queries/02_semantic_triplet_mining.ipynb` | Compute pairwise Jaccard, mine (anchor, positive, negative) triplets |
| `data/successful_queries/successful_queries_dbpedia.json` | Queries with at least one result binding |
| `data/semantic_triplets/semantic_triplets_dbpedia.json` | Mined triplets used for training |
| `data/wikidata_queries/` | (Planned) Wikidata equivalent of above notebooks |

---

## src/

Shared Python library imported by all notebooks.

| File | Class / Purpose |
|------|----------------|
| `config.py` | `Config` — all hyperparameters and paths (edit here to change settings) |
| `sparql_graph_builder.py` | `SPARQLGraphBuilder` — parses a SPARQL string into a labelled NetworkX graph |
| `hybrid_encoder.py` | `HybridSPARQLEncoder` — CodeBERT + GATv2 model, returns L2-normalised embeddings |
| `triplet_dataset.py` | `SemanticSPARQLDataset` — PyTorch Dataset for (anchor, positive, negative) triplets |
| `dataset.py` | Helper dataset utilities |
| `embedder.py` | `SPARQLEmbedder` — high-level inference wrapper; load checkpoint and call `.encode(query)` |
| `requirements.txt` | Pinned Python dependencies |

---

## training/

| Path | Description |
|------|-------------|
| `training/train_dbpedia.ipynb` | Load triplets, build splits, train HybridSPARQLEncoder with in-batch InfoNCE |
| `training/splits/` | Train / valid / test JSON splits (written by the training notebook) |

The notebook saves the trained model to `checkpoints/triplet_model_semantic-<epoch>.pt`.

---

## evaluation/

| Path | Description |
|------|-------------|
| `evaluation/evaluate_dbpedia.ipynb` | Load test split, run model + baselines, compute Jaccard@k / nDCG@k / P/R/F1@k |
| `evaluation/results/` | Output tables saved as CSV and images |

Baselines evaluated: Qwen3-Embedding, MiniLM, E5, MPNet, TF-IDF.

---

## checkpoints/

Saved model weight files (`.pt`). Produced by `training/train_dbpedia.ipynb` and consumed by `evaluation/evaluate_dbpedia.ipynb` and `src/embedder.py`.

Example: `checkpoints/triplet_model_semantic-20.pt` (epoch 20).

---

## docs/

| File | Description |
|------|-------------|
| `docs/architecture.png` | Pipeline architecture diagram (three-stage: data → encoder → contrastive training) |

---

## Run order

1. `data/dbpedia_queries/01_query_collection.ipynb`
2. `data/dbpedia_queries/02_semantic_triplet_mining.ipynb`
3. `training/train_dbpedia.ipynb`
4. `evaluation/evaluate_dbpedia.ipynb`

All notebooks must be run from the **project root** so that relative paths (`../data`, `../checkpoints`, etc.) resolve correctly.
