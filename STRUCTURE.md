# SPARQLCE – Folder structure

## Overview

- **LSQ_queries/** – Data and notebooks for DBpedia (and later Wikidata) pipeline.
- **src/** – Shared Python code (unchanged).
- **Train/** – Training notebooks (DBpedia, later Wikidata).
- **Evaluation/** – Evaluation notebooks (DBpedia, later Wikidata).
- **checkpoints/** – Saved model weights (used by both Train and Evaluation).

---

## LSQ_queries/

| Item | Purpose |
|------|--------|
| **successful_queries/** | Query JSONs: put `successful_queries_dbpedia.json` here (from step 1). Later add `successful_queries_wikidata.json`. |
| **semantic_triplets/** | Semantic triplets JSONs used for training: `semantic_triplets_dbpedia.json` (and later `semantic_triplets_wikidata.json`). |
| **dbpedia_queries/** | DBpedia pipeline notebooks (run in order). |
| ↳ **01_dbpedia_query_collection.ipynb** | Step 1: Fetch and filter queries → writes `../successful_queries/successful_queries_dbpedia.json`. |
| ↳ **02_dbpedia_semantic_triplets.ipynb** | Step 2: Reads from `../successful_queries/`, builds triplets → writes `../semantic_triplets/semantic_triplets_dbpedia.json`. (Later: add a similar folder/notebooks for Wikidata.) |

Run order: **01_dbpedia_query_collection** → **02_dbpedia_semantic_triplets**.

---

## Train/

| File | Purpose |
|------|--------|
| **Train_dbpedia.ipynb** | Trains on `../LSQ_queries/semantic_triplets/semantic_triplets_dbpedia.json`; saves splits to `splits/`, model to `../checkpoints/`. |
| **Train_wikidata.ipynb** | (Later) Same idea for Wikidata. |
| **splits/** | Train/valid/test indices (e.g. `semantic_triplet_splits_dbpedia.json`). |

Run from the project root or with working directory such that `../LSQ_queries` and `../checkpoints` resolve correctly.

---

## Evaluation/

| File | Purpose |
|------|--------|
| **Evaluation_dbpedia.ipynb** | Evaluates using `../LSQ_queries/semantic_triplets/`, `../Train/splits/`, and `../checkpoints/`. |
| **Evaluation_wikidata.ipynb** | (Later) Same for Wikidata. |

---

## Quick reference

- **Successful queries** (input to triplets): `LSQ_queries/successful_queries/successful_queries_dbpedia.json` (and later wikidata).
- **Semantic triplets** (input to training): `LSQ_queries/semantic_triplets/semantic_triplets_dbpedia.json` (and later wikidata).
- **Splits**: `Train/splits/semantic_triplet_splits_dbpedia.json`.
- **Checkpoint**: `checkpoints/triplet_model_semantic-20.pt`.
