# SPARQL Dataset Sampling Pipeline

This directory contains **the pipeline for sampling and filtering unique SPARQL SELECT queries** from the LSQ dataset, and for saving those that return meaningful results from DBpedia.


## Pipeline Notebooks

- `sample_queries.ipynb`:  
  *Fetches SELECT queries from LSQ, removes duplicates, saves to CSV (`unique_select_queries.csv`).*

- `query-DBpedia-and-filter.ipynb`:  
  *Loads unique queries, sends them to DBpedia, saves those with non-empty results to `queries_with_answers.json`.*

## Usage

1. Open `Sampled-LSQ-Dataset/collecting-queries-and-preprocessing.ipynb`.
2. Set SAMPLE_SIZE (or leave sampling defaults) and run the fetch + DBpedia cells to generate `queries_with_answers_sampled.json`.
3. Open `Sampled-LSQ-Dataset/query-similariy-analysis.ipynb` and run cells; it will auto-detect the JSON and run similarity steps. 
4. 

**Requirements:**
- Python 3.8+
- Jupyter Notebook
- Libraries:
  - pandas, requests, tqdm, SPARQLWrapper, sentence-transformers, scikit-learn, torch, ijson, urllib3
- Install:
```bash
pip install pandas requests tqdm SPARQLWrapper sentence-transformers scikit-learn torch ijson urllib3
```
