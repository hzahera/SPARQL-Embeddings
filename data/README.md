# Data Folder Overview

This folder contains the datasets used by the SPARQLCE research pipeline.

## Files and structure

- `binary_classifier_dataset/`
  - `dbpedia_train.json`
  - `dbpedia_val.json`
  - `wikidata_train.json`
  - `wikidata_val.json`
  - `dbpedia_DATASET_CARD.txt`
  - `wikidata_DATASET_CARD.txt`

  These are binary classification datasets of the form:

  ```json
  {
    "anchor_query": "...",
    "query": "...",
    "label": 0 or 1
  }
  ```

  - `dbpedia_train.json` contains 62,954 examples (31,477 positives, 31,477 negatives)
  - `dbpedia_val.json` contains 8,194 examples (4,097 positives, 4,097 negatives)
  - `wikidata_train.json` contains 97,138 examples (48,569 positives, 48,569 negatives)
  - `wikidata_val.json` contains 12,366 examples (6,183 positives, 6,183 negatives)

- `hard_triplets_tfidf_dbpedia.json`
- `hard_triplets_tfidf_wikidata.json`

  These are triplet-style datasets of the form:

  ```json
  {
    "anchor_query": "...",
    "positive_queries": ["...", "..."],
    "negative_queries": ["...", "..."]
  }
  ```

  Summary statistics:
  - `hard_triplets_tfidf_dbpedia.json`: 337 anchor groups
    - positives per anchor: min 3, max 54, mean ~9.43
    - negatives per anchor: min 10, max 20, mean ~19.97
  - `hard_triplets_tfidf_wikidata.json`: 186 anchor groups
    - positives per anchor: min 3, max 55, mean ~11.99
    - negatives per anchor: min 0, max 20, mean ~18.39

- `jaccard_triplets_dbpedia.zip`
- `jaccard_triplets_wikidata.zip`

  Archived mined triplet datasets produced by the triplet mining notebooks. These are raw triplet stores that can be unzipped for use.

- `splits_dbpedia.json`
- `splits_wikidata.json`

  Each file is a dictionary with keys:
  - `seed`
  - `train_idx`
  - `valid_idx`
  - `test_idx`

  These define the train/validation/test split indices for the mined triplet data.

- `sparql2text_dataset.zip`

  Auxiliary seq2seq dataset used for SPARQL-to-text baseline experiments.

## Notes

- The `binary_classifier_dataset` folder is the binary-pair dataset format.
- The `hard_triplets_tfidf_*.json` files are grouped triplets for hard-negative mining.
- The `.zip` files are compressed dataset archives and are not inspected directly here.
