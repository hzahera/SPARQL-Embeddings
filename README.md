# SPARQL Embeddings with Contrastive Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository implements contrastive learning for SPARQL query embeddings using the LSQ dataset. Our approach learns vector representations where queries with similar results are close in embedding space while dissimilar queries are separated.

## Key Features
- 🧠 Contrastive learning framework for SPARQL queries
- 🔍 Multiple SPARQL encoders (e.g., per Knowledge Graphs such as DBpedia or Wikidata)
- 📊 Preprocessing pipeline for LSQ dataset
- 📈 Evaluation on query similarity and caching tasks
- ⚡ Optimized for large-scale SPARQL processing

---

## Preprocessing & Similarity Pipeline

- Loads multiple CSV files containing SPARQL queries from LSQ.
- Filters out non-SELECT queries (ASK, CONSTRUCT, DESCRIBE).
- Removes duplicate SELECT queries (case and whitespace insensitive).
- Saves unique SELECT queries to a new CSV file.
- Executes each query against the DBpedia SPARQL endpoint.
- Collects queries with non-empty results and saves them to a JSON file.
- Computes Jaccard similarity between result sets to find similar queries.
- Saves pairs of queries with high similarity to a JSON file.

### Usage Example

```bash
python preprocess.py \
  --input_dir ./lsq_raw \
  --output_dir ./lsq_processed \
  --endpoint http://dbpedia.org/sparql \
  --max_queries 100000
```

```bash
python similarity.py \
  --input_csv ./lsq_processed/combined_unique_select_queries.csv \
  --endpoint https://dbpedia.data.dice-research.org/sparql \
  --output_json ./lsq_processed/jaccard_similarities_optimized.json
```

---

## DONE
[*] `preprocess.py`: Query preprocessing and filtering script

## TODO
- [ ] `similarity.py`: Query similarity extraction script
- [ ] `model/encoders.py`: Core SPARQL encoder models
- [ ] `model/losses.py`: Contrastive loss functions
- [ ] `evaluation/`: Evaluation scripts
- [ ] `trained_models/`: Pretrained weights
- [ ] Query clustering based on result set similarity
---

## Installation

### Prerequisites
- Python 3.8+
- SPARQL endpoint (Virtuoso/Blazegraph) or pre-loaded RDF dataset
- 8GB+ RAM (16GB recommended)

### Setup
```bash
# Clone repository
git clone https://github.com/your-username/sparql-embeddings.git
cd sparql-embeddings

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation
1. Download LSQ Dataset from LSQ Benchmark

2. Preprocess queries:
```bash 
python preprocess.py \
  --input_dir ./lsq_raw \
  --output_dir ./lsq_processed \
  --endpoint http://dbpedia.org/sparql \
  --max_queries 100000
```

**Arguments:**
- `--endpoint`: SPARQL endpoint for result set extraction  
- `--max_queries`: Maximum queries to process (optional)  
- `--min_results`: Minimum results per query (default=5)

**Training:** *(TODO: Not yet implemented)*

Train embedding model with contrastive loss:

```bash
python train.py \
  --data_dir ./lsq_processed \
  --model_type transformer \
  --batch_size 256 \
  --epochs 100 \
  --output_dir ./models/transformer
```

Key arguments:

--`model_type`: Encoder architecture (bilstm, transformer, gnn)

--`margin`: Contrastive margin (default=1.0)

--`temperature`: NT-Xent temperature

**Evaluation:** *(TODO: Not yet implemented)*

Evaluate trained embeddings:
```bash
python evaluate.py \
  --model_path ./models/transformer/best_model.pt \
  --eval_data ./lsq_processed/test.csv \
  --task retrieval
```

**Available tasks:**
* `retrieval`: k-NN query retrieval (Recall@k, MRR)
* `clustering`: Cluster purity analysis
* `caching`: Query caching simulation

**Pretrained Models** *(TODO: Not yet implemented)*

Download pretrained embeddings:
```bash
wget https://example.com/models/bilstm_base.pt
wget https://example.com/models/transformer_large.pt
```

**Load in Python:** *(TODO: Not yet implemented)*
```python
import torch
from model import SPARQLEncoder

model = SPARQLEncoder(model_type='transformer')
model.load_state_dict(torch.load('transformer_large.pt'))
embedding = model.encode("SELECT ?city WHERE { ?city a dbo:City }")
```

## Directory Structure ##
```bash
├── data/               # Processed datasets
├── docs/               # Evaluation visuals
├── lsq_processing/     # Dataset tools
│   ├── preprocess.py   # [TODO] Not yet implemented
│   └── similarity.py   # [TODO] Not yet implemented
├── model/              # Core models
│   ├── encoders.py     # [TODO] Not yet implemented
│   └── losses.py       # [TODO] Not yet implemented
├── evaluation/         # Evaluation scripts [TODO]
├── trained_models/     # Pretrained weights [TODO]
├── requirements.txt
└── README.md
```