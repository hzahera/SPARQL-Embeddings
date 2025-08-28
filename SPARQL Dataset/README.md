# SPARQL Dataset Notebooks

This folder contains Jupyter notebooks for processing, analyzing, and comparing SPARQL queries and their results. The workflow is designed to help you clean, collect, and analyze SPARQL queries—especially for tasks involving DBpedia and query similarity.


## Contents

- **SPARQL Query Collection & Preprocessing.ipynb**
- **Query Similarity Analysis.ipynb**


## 1. SPARQL Query Collection & Preprocessing

**Purpose:**  
This notebook is responsible for collecting and preparing SPARQL queries for analysis. It performs the following steps:

- **Loads SPARQL queries** (typically from a CSV file, e.g., `combined_unique_select_queries.csv`).
- **Queries the DBpedia endpoint** for each SPARQL query, collecting the results.
- **Filters out unsuccessful queries** (queries that return no results or fail).
- **Saves all successful queries and their results** to a JSON file (`successful_queries.json`).
- **Provides summary statistics** about the collection process.

**Typical Workflow:**
1. Load a list of unique SELECT queries.
2. Send each query to the DBpedia SPARQL endpoint.
3. Store only those queries that return valid results.
4. Output is used as input for the similarity analysis notebook.


## 2. Query Similarity Analysis

**Purpose:**  
This notebook analyzes the similarity between the results of different SPARQL queries. It includes:

- **Loading the results** from `successful_queries.json`.
- **Extracting result sets** for each query.
- **Computing Jaccard similarity** between result sets to find queries with overlapping answers.
- **Computing cosine similarity** between query texts using sentence embeddings (e.g., with Sentence Transformers).
- **Saving the similarity results** to JSON files for further inspection or downstream tasks.
- **Providing tools to inspect and interpret** the most similar queries.

**Typical Workflow:**
1. Load successful queries and their results.
2. For each query, compare its results to others using Jaccard similarity.
3. For similar queries, compute cosine similarity between their query texts.
4. Save and inspect the most similar query pairs.


## How to Use

1. **Preprocessing:**  
   - (Optional but recommended) Use a notebook like `filter_unique_select_queries.ipynb` to clean and deduplicate your raw SPARQL queries before starting this workflow.
2. **Collection:**  
   - Run `SPARQL Query Collection & Preprocessing.ipynb` to fetch results from DBpedia and save successful queries.
3. **Similarity Analysis:**  
   - Run `Query Similarity Analysis.ipynb` to compute and analyze similarities between queries.


## Requirements

- Python 3.8+
- Jupyter Notebook
- pandas
- requests
- tqdm
- scikit-learn
- sentence-transformers
- torch
- ijson
- SPARQLWrapper

Install requirements with:
```bash
pip install pandas requests tqdm scikit-learn sentence-transformers torch ijson SPARQLWrapper
```


## Notes

- The workflow assumes you have a CSV file of SPARQL queries (see preprocessing).
- The DBpedia endpoint can rate-limit or block excessive requests; consider adding delays or batching.
- The similarity analysis can be computationally intensive for large datasets; multiprocessing is used for efficiency.

