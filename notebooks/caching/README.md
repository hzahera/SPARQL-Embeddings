# Experiment: Semantic Query Caching Simulation (`tab:caching`)

**Status: run complete.** `caching_simulation_dbpedia.ipynb` (this folder) has
been executed against the SparqlCE checkpoint; results are in the notebook's
saved cell outputs (the cluster-local `results/caching/caching_dbpedia.json`
was not copied into this repo). `tab:caching` in
`sparql-embeddings-latex/Sections/experiments-v2.tex` is filled with the
$(K{=}100,\theta{=}0.8)$ operating point, the full $3\times3$ grid is in
`tab:caching_full` in `Sections/appendix.tex`, and the `\todo` markers for
this experiment are resolved.

## Goal

Demonstrate the downstream application claimed in §"Downstream Application:
Semantic Query Caching": embeddings trained on answer-set overlap make a
semantic cache *reliable* — a cache hit returns a result that actually matches
what the incoming query would have returned. Text-based embeddings produce
hits that look similar but return wrong results (low result quality).

## Protocol (as committed to in the paper — do not deviate)

- **Query stream.** A held-out portion of the SQuARE corpus, replayed as an
  incoming query stream. Use the **test split** (`datasets/splits_dbpedia.json`,
  `test_idx`, 472 anchors) in a fixed shuffled order (`random.seed(42)`).
- **Cache.** Stores the embeddings (and results) of the **K most recent**
  queries — FIFO eviction on insertion order. Convention: on a hit, the
  incoming query is **not** inserted (its result was never executed); on a
  miss, the query is executed and inserted, evicting the oldest entry if the
  cache is full.
- **Hit rule.** For each incoming query, compute cosine similarity against all
  cached embeddings (all embeddings L2-normalised, so dot product = cosine).
  If the maximum similarity ≥ θ → **hit**: return that cached entry's result.
  Otherwise → **miss**: "execute" the query (look up its true answer set) and
  insert it.
- **Grid.** K ∈ {50, 100, 200} × θ ∈ {0.7, 0.8, 0.9} — 9 configurations per
  model, same stream order for every model and configuration.
- **Models** (the four named in the paper text):

  | Model | How to embed |
  | --- | --- |
  | SparqlCE | `sparqlce/models/final_model_mrr_0.4487.pt` via `SPARQLEmbedder` (same wrapper as the 07 alignment notebooks) |
  | TF-IDF | `sklearn.TfidfVectorizer` fitted on the **train-split** query texts, L2-normalised vectors |
  | E5-base-v2 | `sentence-transformers`, `intfloat/e5-base-v2`, `"query: "` prefix, `normalize_embeddings=True` |
  | Qwen3-Embedding | `Qwen/Qwen3-Embedding-0.6B`, zero-shot, normalised |

## Metrics

Per (model, K, θ):

1. **Hit rate** = hits / stream length.
2. **Result quality** = mean answer-set Jaccard between the *returned* (cached)
   result and the incoming query's *true* result, averaged **over hits**.
   `J(A, B) = |A ∩ B| / |A ∪ B|`, with rows compared as canonical strings
   exactly as stored in the result-sets file.

Report both; the paper's story is the trade-off — text encoders can buy hit
rate only by returning wrong results, SparqlCE keeps quality high.

## Data — everything needed is already in the repo

- `datasets/result_sets_dbpedia.json` — true answer sets for 1,677 queries
  (dict: query string → list of canonical result-row strings). This is the
  "execution oracle": no live endpoint needed.
- `datasets/splits_dbpedia.json` — fixed train/val/test indices (seed 42).
- `datasets/dbpedia-triplets-dataset.json` — maps split indices to anchor
  query strings.
- **Skip rule:** stream queries with no entry (or an empty result set) in the
  result-sets file are dropped from the stream, not counted as misses. Log how
  many were dropped.

The only external requirement is the SparqlCE checkpoint (not in git — see the
Figure 3 runbook, same file).

## Implementation sketch

```python
def simulate(stream, embeddings, results, K, theta):
    cache = deque(maxlen=K)          # entries: (embedding, result_set)
    hits, qualities = 0, []
    for q in stream:
        e_q, R_true = embeddings[q], results[q]
        if cache:
            sims = np.stack([e for e, _ in cache]) @ e_q
            best = int(np.argmax(sims))
            if sims[best] >= theta:
                hits += 1
                R_cached = cache[best][1]
                qualities.append(jaccard(R_cached, R_true))
                continue                     # hit: do not insert
        cache.append((e_q, R_true))          # miss: "execute" + insert
    return hits / len(stream), float(np.mean(qualities)) if qualities else 0.0
```

Embed the full stream once per model up front; the 4 × 9 grid then runs in
seconds.

## Outputs

1. `results/caching/caching_dbpedia.json` — full grid:
   `{model: {"K=50,theta=0.7": {"hit_rate": ..., "result_quality": ..., "n_hits": ...}, ...}}`
   plus stream length, dropped count, and seed.
2. Printed LaTeX rows for `tab:caching`. The current table has **one row per
   model** with a `(K, θ)` column — recommended fill: report the **K=100,
   θ=0.8** operating point in the main table (middle of both grids), and put
   the full 9-point grid in the appendix or extend the table to 2–3
   representative (K, θ) rows per model if space allows.
3. One headline sentence for the prose at `experiments-v2.tex:303`, e.g.
   *"At K=100, θ=0.8, SparqlCE serves X% of the stream from cache at 0.XX
   result quality, while the best text baseline reaches only 0.XX quality at a
   comparable hit rate."*

## Checklist

- [x] Write `notebooks/caching/caching_simulation_dbpedia.ipynb` implementing this spec
- [x] Run with SparqlCE checkpoint in place
- [x] Fill `tab:caching` + delete the `\todo` at `experiments-v2.tex:303`
- [ ] (Optional) Wikidata twin — needs `result_sets_wikidata.json`, which is
      **not currently in `datasets/`**; DBpedia alone is sufficient for the
      downstream-application claim
