# Semantic query caching simulation

`caching_simulation_dbpedia.ipynb` (DBpedia) and
`caching_simulation_wikidata.ipynb` (Wikidata) simulate a semantic query cache
to test whether embeddings trained on answer-set overlap make cache hits
*reliable* — a hit should return a result that actually matches what the
incoming query would have returned. Text-based embeddings can produce hits
that look similar but return the wrong results.

## Protocol

- **Query stream.** The graph's test split (`datasets/splits_{dbpedia,
  wikidata}.json`, `test_idx`: 472 anchors on DBpedia, 1,074 on Wikidata),
  replayed in a fixed shuffled order (`random.seed(42)`).
- **Cache.** Stores the embeddings (and results) of the **K most recent**
  queries — FIFO eviction on insertion order. On a hit, the incoming query is
  not inserted; on a miss, the query is executed and inserted, evicting the
  oldest entry if the cache is full.
- **Hit rule.** For each incoming query, compute cosine similarity against all
  cached embeddings. If the maximum similarity ≥ θ → hit: return that cached
  entry's result. Otherwise → miss: look up the query's true answer set and
  insert it.
- **Grid.** K ∈ {50, 100, 200} × θ ∈ {0.7, 0.8, 0.9}, evaluated for eight
  models: SparqlCE, TF-IDF, BM25, E5-base-v2, Qwen3-Embedding-0.6B,
  Qwen3-Embedding-0.6B (fine-tuned), SBERT/MiniLM, SBERT/MPNet.

## Metrics

Per (model, K, θ):

1. **Hit rate** = hits / stream length.
2. **Result quality** = mean answer-set Jaccard between the returned (cached)
   result and the incoming query's true result, averaged over hits.
3. **Utility** = hit rate × quality, the expected served Jaccard per query —
   reconciles the other two into the single number reported in the paper's
   `fig:caching` (see `scripts/plot_caching_utility.py`).

## Data

- `datasets/result_sets_{dbpedia,wikidata}.json` — true answer sets, used as
  the execution oracle (no live SPARQL endpoint needed). DBpedia's is ~4MB;
  Wikidata's is ~6GB (much longer answer sets per query), so the Wikidata
  notebook streams and filters it with `ijson` instead of loading it whole —
  see that notebook's data cell.
- `datasets/splits_{dbpedia,wikidata}.json` — fixed train/val/test indices.
- `datasets/{dbpedia,wikidata}-triplets-dataset.json` — maps split indices to
  anchor query strings.

Running either notebook additionally requires a trained SparqlCE checkpoint
(`sparqlce/models/`, not tracked in git — see the main
[README](../../README.md)) and, for the fine-tuned Qwen3 baseline, the LoRA
adapter at `scripts/qwen_ft/qwen3_square_{dbpedia,wikidata}/best_adapter`
(produced by `notebooks/finetuning/`).
