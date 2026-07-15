# Semantic query caching simulation

`caching_simulation_dbpedia.ipynb` simulates a semantic query cache to test
whether embeddings trained on answer-set overlap make cache hits *reliable* —
a hit should return a result that actually matches what the incoming query
would have returned. Text-based embeddings can produce hits that look similar
but return the wrong results.

## Protocol

- **Query stream.** The DBpedia test split (`datasets/splits_dbpedia.json`,
  `test_idx`, 472 anchors), replayed in a fixed shuffled order
  (`random.seed(42)`).
- **Cache.** Stores the embeddings (and results) of the **K most recent**
  queries — FIFO eviction on insertion order. On a hit, the incoming query is
  not inserted; on a miss, the query is executed and inserted, evicting the
  oldest entry if the cache is full.
- **Hit rule.** For each incoming query, compute cosine similarity against all
  cached embeddings. If the maximum similarity ≥ θ → hit: return that cached
  entry's result. Otherwise → miss: look up the query's true answer set and
  insert it.
- **Grid.** K ∈ {50, 100, 200} × θ ∈ {0.7, 0.8, 0.9}, evaluated for four
  models: SparqlCE, TF-IDF, E5-base-v2, and Qwen3-Embedding.

## Metrics

Per (model, K, θ):

1. **Hit rate** = hits / stream length.
2. **Result quality** = mean answer-set Jaccard between the returned (cached)
   result and the incoming query's true result, averaged over hits.

## Data

- `datasets/result_sets_dbpedia.json` — true answer sets for 1,677 queries,
  used as the execution oracle (no live SPARQL endpoint needed).
- `datasets/splits_dbpedia.json` — fixed train/val/test indices.
- `datasets/dbpedia-triplets-dataset.json` — maps split indices to anchor
  query strings.

Running the notebook additionally requires a trained SparqlCE checkpoint
(`sparqlce/models/`, not tracked in git — see the main [README](../../README.md)).
