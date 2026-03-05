# SPARQLCE — SPARQL Query Embeddings with Semantic Triplets

A machine learning pipeline for learning dense embeddings of SPARQL queries and evaluating them on semantic similarity. The project uses **DBpedia** (and is structured for **Wikidata** later): it collects queries from the LSQ endpoint, builds **semantic triplets** from result-set Jaccard similarity, trains a **Hybrid SPARQL encoder** (CodeBERT + GNN) with in-batch negative contrastive learning, and evaluates retrieval against several baselines.

---

## Overview

**SPARQLCE** trains a neural encoder that maps SPARQL queries to vectors so that **semantically similar queries** (similar result sets over a SPARQL endpoint) are close in the embedding space. Relevance is defined by **result-set Jaccard similarity**: two queries are similar if they return overlapping result bindings when executed. The pipeline:

1. **Collects** SPARQL queries from the [LSQ](http://lsq.aksw.org/) endpoint (DBpedia-backed) and keeps only queries that return results.
2. **Builds semantic triplets**: for each anchor query, positives are queries with high Jaccard (e.g. ≥ 0.8) and negatives with low Jaccard (e.g. ≤ 0.1).
3. **Trains** a **Hybrid SPARQL encoder** (CodeBERT + GAT) on these triplets using **in-batch negative** InfoNCE loss (one positive per anchor; other batch items as negatives).
4. **Evaluates** retrieval: given an anchor, rank a pool (positives + negatives) by embedding similarity; report graded metrics (avg Jaccard@k, nDCG@k) and binary P/R/F1@k using Jaccard ≥ 0.8 as relevance.

Baselines (Qwen, MiniLM, E5, MPNet, TF-IDF) are compared in the same evaluation setup.

---

## Project Structure

| Directory / File | Description |
|------------------|-------------|
| **LSQ_queries/** | Data and notebooks for the query/triplet pipeline. |
| ↳ **successful_queries/** | JSON files of successful queries (e.g. `successful_queries_dbpedia.json`). |
| ↳ **semantic_triplets/** | Semantic triplet datasets (e.g. `semantic_triplets_dbpedia.json`) used for training. |
| ↳ **dbpedia_queries/** | Notebooks: `01_dbpedia_query_collection.ipynb`, `02_dbpedia_semantic_triplets.ipynb`. |
| **src/** | Shared Python code: config, graph builder, model, dataset, inference. |
| **Train/** | Training notebooks (e.g. `Train_dbpedia.ipynb`); writes splits and checkpoints. |
| **Evaluation/** | Evaluation notebook (e.g. `Evaluation_dbpedia.ipynb`); compares model vs baselines, saves results. |
| **checkpoints/** | Saved model weights (e.g. `triplet_model_semantic-20.pt`). |
| **Results/** | Evaluation outputs (tables as images and CSV). |
| **STRUCTURE.md** | Detailed folder and file reference. |

---

## Pipeline (Run Order)

Run all notebooks from the **project root** so that paths like `../LSQ_queries`, `../checkpoints`, and `../Train` resolve correctly.

1. **LSQ_queries/dbpedia_queries/01_dbpedia_query_collection.ipynb**  
   Fetches SPARQL queries from LSQ (DBpedia), runs them against the DBpedia endpoint, and saves those that return results to `LSQ_queries/successful_queries/successful_queries_dbpedia.json`.

2. **LSQ_queries/dbpedia_queries/02_dbpedia_semantic_triplets.ipynb**  
   Reads successful queries, executes them to get result sets, and builds semantic triplets (anchor, positives, negatives) using Jaccard thresholds. Writes `LSQ_queries/semantic_triplets/semantic_triplets_dbpedia.json`.

3. **Train/Train_dbpedia.ipynb**  
   Loads the semantic triplets, builds train/valid/test splits (saved under `Train/splits/`), and trains the Hybrid SPARQL encoder with in-batch negatives. Saves the model to `checkpoints/triplet_model_semantic-20.pt`.

4. **Evaluation/Evaluation_dbpedia.ipynb**  
   Loads the test split and triplets, builds a unique set of queries, runs SPARQL and encodes with the trained model and baselines, then ranks each pool and computes metrics. Saves tables (e.g. as images and CSV) under `Results/`.

---

## Model and Baselines

- **HybridSPARQLEncoder (ours):** CodeBERT for token embeddings + GAT over the SPARQL query graph; projection to a fixed-dimension L2-normalized vector. Trained with in-batch InfoNCE and optional Jaccard regression.
- **Baselines:** Qwen3-Embedding, MiniLM, E5, MPNet (sentence-transformers), and TF-IDF (word 1–2 grams). All produce one vector per query; ranking is by cosine similarity (or dot product on L2-normalized vectors).

---

## Evaluation Metrics

- **Graded relevance:** Result-set Jaccard is computed between anchor and each query in the pool (using cached SPARQL results).  
  - **avg Jaccard@k:** Mean Jaccard of the top-k retrieved.  
  - **nDCG@k:** Normalized DCG using these Jaccard scores.
- **Binary relevance:** Queries with Jaccard ≥ 0.8 are “relevant”.  
  - **P/R/F1@k:** Precision, recall, and F1 at k for this binary relevance.

Evaluation uses a **unique set** of all queries that appear in the test set; SPARQL is run once per query and embeddings are computed once per model.

---

## Prerequisites and Setup

- **Python** 3.10+ (or as required by the notebooks).
- **CUDA** optional but recommended for training.

Install dependencies (e.g. from project root):

```bash
pip install -r src/requirements.txt
```

Additional packages used in notebooks (install if missing): `pandas`, `requests`, `ijson`, `SPARQLWrapper`, `sentence-transformers`, `scikit-learn`, `matplotlib`, `IPython`.

---

## Configuration

- **Training:** `src/config.py` defines model (CodeBERT), dimensions, batch size, learning rate, epochs, and temperature. The Train notebook uses in-batch negatives and a fixed batch size (e.g. 16).
- **Evaluation:** Paths (semantic triplets, splits, checkpoint), baseline model names, `TOPKS`, `BATCH`, and `JACCARD_BINARY_THRESHOLD` are set in the Evaluation notebook.

---

## References

- **LSQ:** [Linked SPARQL Queries](http://lsq.aksw.org/) — endpoint and vocabulary for query metadata.
- **DBpedia:** SPARQL endpoint used for executing queries and defining result-set similarity.
- **CodeBERT:** Used as the text encoder in the hybrid model (`microsoft/codebert-base`).

---
