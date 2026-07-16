# Query log compaction (coreset selection)

`query_log_compaction_dbpedia.ipynb` tests whether embeddings trained on
answer-set overlap can compress a query log into a small representative
subset without losing what the log *means* — i.e. every discarded query
should still be well-represented by some kept query's actual results, not
just its wording. This replaces the semantic-caching simulation
(`notebooks/caching/`) as the paper's downstream-application experiment; see
that notebook's README and this one's intro cell for why (shared-θ fairness
across differently-calibrated embedding spaces, and a sequential/
path-dependent simulation that amplified GPU non-determinism run to run).

## Protocol

- **Corpus.** The RQ1 DBpedia global corpus (`datasets/dbpedia_testset_RQ1.json`,
  1,677 queries — the same corpus used for `tab:dbpedia_ranking_new`).
- **Selection.** Deterministic agglomerative clustering (average linkage)
  over each model's own cosine-distance matrix, cut to *k* clusters. Each
  cluster's medoid (minimum total in-cluster distance) becomes the
  representative for every query in that cluster. No random initialization
  (unlike k-means/k-medoids), so the same embeddings always yield the same
  coreset.
- **Budget grid.** k/N ∈ {50%, 30%, 20%, 10%, 5%} of the corpus, identical
  for every model — comparison is at **equal budget**, not a shared
  absolute similarity threshold, so no model is favored or penalized by its
  own similarity scale.
- **Models.** Same roster as the caching experiment: SparqlCE, TF-IDF, BM25,
  E5-base-v2, Qwen3-Embedding-0.6B (zero-shot and fine-tuned), SBERT/MiniLM,
  SBERT/MPNet.

## Metrics

Per (model, k):

1. **Mean fidelity** = mean answer-set Jaccard between each query's true
   result set and its cluster medoid's true result set, averaged over the
   whole corpus.
2. **Fraction high-fidelity** = fraction of queries with Jaccard ≥ 0.8
   against their medoid (the same positive-construction threshold used
   throughout the paper).

Selection only ever sees embeddings; oracle answer sets are used exclusively
to score fidelity after clustering, never to guide it.

## Data

- `datasets/dbpedia_testset_RQ1.json` — the corpus.
- `datasets/result_sets_dbpedia.json` — true answer sets (verified to cover
  100% of the corpus before this notebook was written).

Running the notebook additionally requires a trained SparqlCE checkpoint
(`sparqlce/models/final_model_mrr_0.4487.pt`) and the fine-tuned Qwen3 LoRA
adapter (`scripts/qwen_ft/qwen3_square_dbpedia/best_adapter`), neither
tracked in git — see the main [README](../../README.md).

## Determinism

Same fix as `notebooks/caching/caching_simulation_dbpedia.ipynb`:
`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG`, and a
CPU fallback for SparqlCE's GATv2 aggregation if that op has no
deterministic CUDA kernel. Additionally, agglomerative clustering itself
(unlike k-means/k-medoids) has no random initialization, so the coreset
selection step adds no non-determinism on top of the embeddings.
