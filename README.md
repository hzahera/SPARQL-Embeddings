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
├── scripts/                    # Dataset generation and robustness evaluation scripts
│   ├── training/               # Training scripts
│   │   └── generate_training_triplets.py
│   └── robustness/             # Robustness evaluation scripts
│       └── robustness_dataset_generation.py
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

3. Unzip the raw triplet archives if needed:

```bash
cd data
unzip jaccard_triplets_dbpedia.zip
unzip jaccard_triplets_wikidata.zip
cd ..
```

4. Start Jupyter from the project root:

```bash
jupyter lab
```

## Using the library

```python
from sparqlce.embedder import SPARQLEmbedder

embedder = SPARQLEmbedder("checkpoints/triplet_model_semantic-20.pt")
query = "SELECT ?person WHERE { ?person a <dbo:Scientist> . ?person <dbo:birthPlace> ?place }"
embedding = embedder.encode(query)
print(embedding.shape)
```

## Core modules

- `sparqlce/config.py` — consolidated hyperparameters, paths, and device selection
- `sparqlce/graph_builder.py` — builds labelled NetworkX graphs from SPARQL queries
- `sparqlce/encoder.py` — hybrid CodeBERT + GATv2 query encoder
- `sparqlce/dataset.py` — triplet dataset loader and PyG collate helpers
- `sparqlce/embedder.py` — inference wrapper for trained checkpoints
- `sparqlce/sparql_utils.py` — SPARQL endpoint utilities and caching
- `sparqlce/eval_utils.py` — retrieval metrics and baseline evaluation helpers

## Scripts

- `scripts/robustness_dataset_generation.py` — build perturbation datasets for robustness evaluation
- `scripts/robustness_eval.py` — compute robustness metrics and cosine similarity for baselines and SPARQLECE

## Data overview

- `data/jaccard_triplets_dbpedia.zip` and `data/jaccard_triplets_wikidata.zip` — mined SPARQL triplets
- `data/splits_dbpedia.json`, `data/splits_wikidata.json` — train/val/test splits
- `data/hard_triplets_tfidf_dbpedia.json`, `data/hard_triplets_tfidf_wikidata.json` — hard negatives mined with TF-IDF
- `data/binary_classifier_dataset/` — binary classifier dataset
- `data/sparql2text_dataset.zip` — auxiliary seq2seq baseline dataset

See `data/README.md` for detailed dataset statistics and file structure.

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
