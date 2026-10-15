# Non-neural baselines

Code that produces the non-neural retrieval baselines reported in the paper.

## Methods

| Method | What it scores |
|---|---|
| `random` | random order (averaged over 50 runs) |
| `tfidf` | cosine of TF-IDF vectors over SPARQL tokens (variable names masked) |
| `bm25` | BM25 (k1 = 1.2, b = 0.75) over the same tokens |
| `normalized_match` | 1 if the variable-renamed rdflib algebra is identical, else 0 |
| `algebra_similarity` | weighted Jaccard over algebra node labels and parent>child edges |
| `query_form` | negative distance between standardized query-form features (arity, triple patterns, OPTIONAL, UNION, FILTER, ...) |

All statistics (IDF, feature scaling) are fitted on the **training split only**. Nothing is tuned.
Ties are broken at random and averaged over several runs.

## Protocols

| Protocol | Candidates | Relevance |
|---|---|---|
| `A` sampled | each anchor's stored candidates (up to 20 positives + 20 negatives) | stored answer Jaccard |
| `B` code | all test-split queries (same as `evaluation.rank_record_global`) | stored Jaccard; unstored = non-relevant |
| `C` full | all test-split queries | complete pair labels from `*-test-rq3.json` (J >= 0.8 relevant) |

Metrics: Hits@1, MRR@5, nDCG@5, MRR, nDCG@10 (from `src/result_aware/metrics.py`).

## Run

From the repository root:

```bash
pip install -r scripts/baselines/requirements.txt
bash scripts/baselines/run_baselines.sh            # both graphs
bash scripts/baselines/run_baselines.sh dbpedia    # one graph
```

Or directly:

```bash
PYTHONHASHSEED=0 python scripts/baselines/nonneural.py . dbpedia  scripts/baselines/results_dbpedia.json
PYTHONHASHSEED=0 python scripts/baselines/nonneural.py . wikidata scripts/baselines/results_wikidata.json
```

Run time: about 30 s for DBpedia on a laptop; Wikidata takes longer.

## Inputs

- `datasets/<graph>-splits/<graph>-{train,test,test-rq3}.json`
- `scripts/audit/audit_<graph>_perquery.json`: per-query features (algebra hash, arity, triple patterns)
  made by `scripts/audit/audit_splits.py`. These files are committed so the paper numbers
  reproduce exactly.

## Reproducibility note

The algebra hash in `audit_splits.py` still depends on Python's hash seed (rdflib builds some
parts of the algebra from sets). Running `audit_splits.py` again can therefore give slightly
different `normalized_match` numbers. All other methods do not use this hash and are not affected.
To get the paper numbers, use the committed `audit_<graph>_perquery.json` files.
Check (DBpedia, Python 3.13, rdflib 7.6.0): with the committed audit file the output matches
`results_dbpedia.json` exactly.

## Outputs

`results_<graph>.json` holds one entry per `<protocol>:<method>`, plus:

- `normalized_match_recovers_positive_pairs_share`: share of RQ3 positive pairs with identical normalized algebra
- `struct_parse_failures`: queries rdflib could not parse (0 on both graphs)
