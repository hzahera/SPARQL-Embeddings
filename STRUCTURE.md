# SPARQLCE — Folder & File Reference

## Top-level layout

```
SPARQL-Embeddings/
├── dataset/                  # All dataset artifacts
├── src/                      # Core Python library
├── training/                 # Training notebooks
├── evaluation/               # Evaluation notebooks & results
├── checkpoints/              # Saved model weights (.pt files)
├── docs/                     # Documentation assets
├── requirements.txt          # Python dependencies (install from here)
├── CITATION.cff
├── STRUCTURE.md              # This file
├── README.md
└── LICENSE
```

---

## dataset/

Holds all raw query logs, mined triplets, and index splits.

```
dataset/
├── dbpedia/
│   ├── 01_query_collection.ipynb     # Fetch queries from LSQ, execute against DBpedia
│   └── 02_triplet_mining.ipynb       # Compute pairwise Jaccard, mine (anchor, pos, neg) triplets
├── wikidata/
│   ├── 01_query_collection.ipynb     # Wikidata equivalent
│   └── 02_triplet_mining.ipynb
├── raw/
│   ├── jaccard_triplets_dbpedia.json         # Mined triplets for DBpedia
│   └── jaccard_triplets_wikidata.json.gz     # Mined triplets for Wikidata (gzipped)
└── splits/
    ├── splits_dbpedia.json               # Train/valid/test index splits
    └── splits_wikidata.json
```

**JSON schema** for each triplet file:
```json
[
  {
    "anchor_query": "SELECT ...",
    "positive_queries": [{"query": "...", "answer_jaccard": 0.9}, ...],
    "negative_queries": [{"query": "...", "answer_jaccard": 0.05}, ...]
  }
]
```

---

## src/

Shared Python library imported by all notebooks.

| File | Class / Purpose |
|---|---|
| `__init__.py` | Exposes public API; enables `from src import SPARQLEmbedder` |
| `config.py` | `Config` — all hyperparameters and dataset paths (edit here first) |
| `sparql_graph_builder.py` | `SPARQLGraphBuilder` — parses a SPARQL string into a labelled NetworkX graph |
| `hybrid_encoder.py` | `HybridSPARQLEncoder` — CodeBERT + GATv2 model, L2-normalised embeddings |
| `triplet_dataset.py` | `SemanticSPARQLDataset` + collate functions — PyTorch Dataset for triplets |
| `embedder.py` | `SPARQLEmbedder` — high-level inference wrapper; load checkpoint and call `.encode(query)` |
| `requirements.txt` | Pinned dependency versions (legacy; canonical file is root `requirements.txt`) |

---

## training/

| Notebook | Description |
|---|---|
| `01_train_dbpedia.ipynb` | Load `dataset/raw/jaccard_triplets_dbpedia.json`, create splits, train HybridSPARQLEncoder |
| `02_train_wikidata.ipynb` | Wikidata equivalent |

The notebook saves the trained model to `checkpoints/triplet_model_semantic-<epoch>.pt`.

---

## evaluation/

| Notebook | Description |
|---|---|
| `01_evaluate_dbpedia.ipynb` | Load test split, run SPARQLCE + baselines, compute Jaccard@k / nDCG@k / P⁄R⁄F1@k |
| `02_evaluate_wikidata.ipynb` | Wikidata equivalent |

Baselines: Qwen3-Embedding, MiniLM, E5, MPNet, TF-IDF.
Results are written to `evaluation/results/` (excluded from version control).

---

## checkpoints/

Saved model weight files (`.pt`). **Not tracked in git** — download from the project's Zenodo/HuggingFace release.

Example: `checkpoints/triplet_model_semantic-20.pt` (epoch 20).

---

## docs/

| File | Description |
|---|---|
| `architecture.png` | Three-stage pipeline diagram: Data → SPARQL-to-Graph Encoding → Contrastive Training |

---

## Run order

All notebooks must be run from the **project root** so that relative paths resolve correctly.

| Step | Notebook | Output |
|---|---|---|
| 1 | `dataset/dbpedia/01_query_collection.ipynb` | `dataset/raw/jaccard_triplets_dbpedia.json` |
| 2 | `dataset/dbpedia/02_triplet_mining.ipynb` | `dataset/raw/jaccard_triplets_dbpedia.json` (enriched with Jaccard scores) |
| 3 | `training/01_train_dbpedia.ipynb` | `checkpoints/triplet_model_semantic-20.pt` + `dataset/splits/splits_dbpedia.json` |
| 4 | `evaluation/01_evaluate_dbpedia.ipynb` | Metric tables in `evaluation/results/` |

Repeat steps 1–4 using the `wikidata/` variants for the Wikidata benchmark.
