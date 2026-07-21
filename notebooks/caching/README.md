# Semantic Query Caching Experiment

## Overview

This experiment evaluates whether query embeddings can support a reliable semantic cache for SPARQL workloads.

A semantic query cache stores previously executed queries together with their answer sets. When a new query arrives, the cache compares its embedding with the embeddings of cached queries. If a sufficiently similar cached query exists, the cache returns that stored answer set instead of executing the incoming query.

The central question is:

> Does a cache hit return an answer set that actually matches what the incoming query would have returned?

This distinction matters because two SPARQL queries may look similar while returning very different results, and two queries may look different while returning nearly identical results. A text-based embedding can therefore produce a high-similarity cache hit that serves the wrong answer set.

The experiment uses precomputed answer sets as an execution oracle, so no live SPARQL endpoint is required.

---

## Research Objective

The experiment tests whether embedding similarity is sufficiently aligned with answer-set overlap for semantic query caching.

For each incoming query, the cache:

1. compares the query against the currently cached queries;
2. selects the single most similar cached query;
3. returns its stored result if the similarity exceeds a threshold;
4. otherwise treats the request as a miss, retrieves the true result from the offline oracle, and inserts the query into the cache.

The primary evaluation focuses on **top-1 caching**: only the most similar cached query may be returned.

---

## Dataset Layout

The repository is expected to follow this structure:

```text
SPARQL-Embeddings/
├── datasets/
│   ├── splits_dbpedia.json
│   ├── splits_wikidata.json
│   ├── dbpedia-triplets-dataset.json
│   ├── wikidata-triplets-dataset.json
│   ├── result_sets_dbpedia.json
│   └── result_sets_wikidata.json
└── notebooks/
    └── caching/
        └── caching_simulation_*.ipynb
```

### Split files

```text
datasets/splits_dbpedia.json
datasets/splits_wikidata.json
```

These files contain the fixed train, validation, and test indices. The caching stream is built from the test split.

The expected test field is one of:

```json
{
  "test_idx": [...]
}
```

or an equivalent field such as `test_indices` or `test`.

### Triplet datasets

```text
datasets/dbpedia-triplets-dataset.json
datasets/wikidata-triplets-dataset.json
```

These files map split indices to anchor SPARQL query strings.

Only the anchor query associated with each test index is used to build the cache stream.

### Result-set oracle

```text
datasets/result_sets_dbpedia.json
datasets/result_sets_wikidata.json
```

These files contain the true answer sets for the SPARQL queries.

They replace live query execution during simulation. On a cache miss, the simulator looks up the incoming query's true answer set in this file.

---

## Query Stream Construction

The query stream is constructed as follows:

1. Load the test indices from the split file.
2. Map each test index to its anchor SPARQL query.
3. Verify that every query has a corresponding answer set in the oracle.
4. Preserve duplicate query occurrences unless the experiment explicitly specifies otherwise.
5. Shuffle the stream once using:

```python
random.seed(42)
```

or, preferably:

```python
rng = random.Random(42)
rng.shuffle(stream)
```

6. Use the exact same shuffled stream for every model and cache configuration.
7. Save the final ordered stream to a manifest file for reproducibility.

Example manifest fields:

```text
stream_position
source_index
query
answer_set_size
```

Queries missing oracle results must not be silently removed. They should either cause the experiment to fail or be reported explicitly in an audit table.

---

## Baselines

The initial baseline set is:

### Dense embedding models

- SBERT / MiniLM  
  `sentence-transformers/all-MiniLM-L6-v2`

- SBERT / MPNet  
  `sentence-transformers/all-mpnet-base-v2`

- Qwen3-Embedding-0.6B, zero-shot  
  `Qwen/Qwen3-Embedding-0.6B`

Dense embeddings must be L2-normalized before cache simulation so that the inner product equals cosine similarity.

### Sparse baselines

- TF-IDF
- BM25-weighted sparse cosine

TF-IDF and BM25 representations should use a SPARQL-aware tokenizer when possible.

The tokenizer should preserve useful SPARQL units such as:

- variables, for example `?person`;
- IRIs, for example `<http://dbpedia.org/resource/Paris>`;
- prefixed names, for example `dbo:birthPlace`;
- literals;
- keywords;
- operators;
- punctuation relevant to query structure.

### BM25 clarification

Raw BM25 scores are not naturally symmetric, normalized, or bounded to the same range as cosine similarity.

For a shared cosine-threshold cache experiment, BM25 should therefore be implemented as:

1. term-count vectors;
2. BM25-style term weighting;
3. L2 normalization;
4. cosine similarity between the resulting sparse vectors.

This condition should be named clearly, for example:

```text
BM25-weighted cosine
```

It should not be described as standard asymmetric BM25 retrieval.

---

## Embedding Preparation

For each model:

1. collect all unique query strings required by the stream;
2. generate one representation per unique query;
3. L2-normalize the representation;
4. cache the representation matrix on disk;
5. map every stream occurrence to the corresponding representation row.

The same query order must be used when saving and loading embeddings.

A digest of the ordered query list should be stored with each embedding cache to prevent accidental reuse with a different stream or query ordering.

Recommended cache metadata:

```text
model identifier
query-list digest
embedding dimension
prompt configuration
normalization status
```

---

## Qwen Zero-Shot Configuration

For a symmetric query-to-query comparison, all SPARQL queries should be embedded in the same way.

A simple zero-shot configuration is:

```python
prompt_name = None
```

This embeds every SPARQL query as a plain string.

Applying a retrieval prompt such as `prompt_name="query"` to all queries changes the experimental condition and should be reported separately.

The prompt configuration must remain fixed across the full cache grid.

---

## Cache State

Each cache entry stores:

```text
query identifier
query embedding or vector row
answer set
insertion position
```

The cache contains at most `K` entries.

The cache is ordered by insertion time.

---

## FIFO Eviction Policy

The cache uses FIFO eviction based on insertion order.

When a miss occurs:

1. execute the incoming query through the offline result-set oracle;
2. insert the incoming query and its result;
3. if the cache is already full, evict the oldest inserted entry.

A cache hit does not change insertion order.

This means the cache is FIFO, not LRU.

---

## Top-1 Hit Rule

For an incoming query \(q\), compute cosine similarity against every cached query:

\[
s(q,c)=\cos(e_q,e_c)
\]

where \(e_q\) and \(e_c\) are normalized query representations.

Let:

\[
c^*=\arg\max_{c\in C}s(q,c)
\]

where \(C\) is the current cache.

The request is a hit when:

\[
s(q,c^*) \geq \theta
\]

On a hit:

- return the answer set stored for \(c^*\);
- do not execute the incoming query;
- do not insert the incoming query;
- do not refresh or reorder the cached entry.

On a miss:

- obtain the true answer set from the oracle;
- insert the incoming query;
- evict the oldest entry if the cache is full.

If the cache is empty, the request is always a miss.

---

## Deterministic Tie-Breaking

If multiple cached entries have exactly the same similarity, the simulator must use a deterministic rule.

Recommended rule:

> Select the earliest matching entry in the current FIFO cache.

With an array-based implementation, this is usually obtained naturally from `argmax`, which returns the first maximum.

The tie-breaking rule must be identical for every model.

---

## Cache Grid

Evaluate every model over:

\[
K \in \{50,100,200\}
\]

and:

\[
\theta \in \{0.7,0.8,0.9\}
\]

This produces nine cache configurations per model.

For five baselines, the full experiment contains:

\[
5 \times 3 \times 3 = 45
\]

configurations.

---

## Answer-Set Canonicalization

Answer sets must be converted into a stable set representation before Jaccard calculation.

A result binding may be represented as:

- a URI string;
- a literal;
- a tuple or list;
- a dictionary of variable bindings;
- a SPARQL JSON binding object.

Each complete answer binding should be converted to a deterministic string, for example using sorted JSON serialization.

Example:

```python
json.dumps(
    binding,
    sort_keys=True,
    ensure_ascii=False,
    separators=(",", ":"),
)
```

The resulting strings form a Python set or `frozenset`.

This preserves complete multi-variable rows rather than comparing individual cells independently.

---

## Answer-Set Jaccard

For an incoming query \(q\) and returned cached query \(c\), define:

\[
J(q,c)=
\frac{|A(q)\cap A(c)|}
{|A(q)\cup A(c)|}
\]

where:

- \(A(q)\) is the incoming query's true answer set;
- \(A(c)\) is the cached query's stored answer set.

Recommended empty-set convention:

```text
Jaccard(empty, empty) = 0
```

This convention must be documented and used consistently across training, evaluation, and caching.

---

## Primary Metrics

Metrics are calculated separately for each:

```text
model
cache capacity K
similarity threshold theta
```

### Hit rate

\[
\text{HitRate}
=
\frac{\text{number of hits}}
{\text{stream length}}
\]

Cold-start misses are included in the denominator.

### Result quality

Let \(H\) be the set of hit positions.

\[
\text{Quality}
=
\frac{1}{|H|}
\sum_{i\in H}J_i
\]

where \(J_i\) is the answer-set Jaccard between the returned cached result and the incoming query's true result.

If there are no hits, quality should be reported as `NaN`, not zero.

### Utility

\[
\text{Utility}
=
\text{HitRate}\times\text{Quality}
\]

Equivalently:

\[
\text{Utility}
=
\frac{1}{N}
\sum_{i\in H}J_i
\]

where \(N\) is the stream length.

The second expression is preferred in code because it remains well-defined when there are no hits.

Utility can be interpreted as:

> The expected answer-set Jaccard served from the cache per incoming query.

This is the main top-1 cache score.

---

## Additional Reliability Metrics

The following metrics are useful for diagnosing unsafe cache hits.

### Reliable-hit precision

For a reliability threshold of \(0.8\):

\[
\text{ReliableHitPrecision}
=
P(J\geq0.8\mid\text{hit})
\]

This uses the number of hits as its denominator.

### Exact-hit precision

\[
\text{ExactHitPrecision}
=
P(J=1\mid\text{hit})
\]

### Wrong-hit rate per query

\[
\text{WrongHitRate}
=
\frac{
|\{i\in H:J_i<0.8\}|
}{N}
\]

This uses the full stream length as its denominator.

### Catastrophic-hit rate per query

\[
\text{CatastrophicHitRate}
=
\frac{
|\{i\in H:J_i=0\}|
}{N}
\]

This measures the fraction of all incoming queries for which the cache returned a completely disjoint answer set.

Metric names should make the denominator explicit.

---

## Per-Query Trace

The simulator should optionally save a trace for each configuration.

Recommended fields:

```text
model
capacity
theta
stream_position
query
cache_size_before
decision
best_similarity
returned_query
hit_jaccard
reliable_hit
exact_hit
catastrophic_hit
inserted
evicted_query
cache_size_after
```

The trace supports:

- debugging the cache state;
- checking oracle alignment;
- inspecting false hits;
- verifying FIFO behavior;
- validating threshold effects;
- manually examining individual query pairs.

---

## Required Sanity Checks

Before reporting any results, verify the following.

### Stream consistency

Every model and cache configuration must use the same stream length and order.

### Count consistency

For every configuration:

\[
\text{hits}+\text{misses}=N
\]

### Utility identity

For configurations with at least one hit:

\[
\text{Utility}
=
\text{HitRate}\times\text{Quality}
\]

up to floating-point tolerance.

### Metric bounds

All rates and Jaccard-based metrics must lie in:

\[
[0,1]
\]

### Vector normalization

Every nonzero query representation must have unit L2 norm.

### Cache capacity

The cache size must never exceed \(K\).

### Hit behavior

A hit must not:

- insert the incoming query;
- refresh the returned cached entry;
- change FIFO ordering.

### Miss behavior

A miss must:

- obtain the oracle result;
- insert the incoming query;
- evict exactly one oldest entry when the cache is full.

---

## Synthetic Protocol Test

Before running real models, test the simulator using a tiny synthetic stream.

The synthetic test should include:

1. an initial miss;
2. a second query whose vector is similar enough to cause a hit;
3. verification that the hit is not inserted;
4. a later miss;
5. an assertion that utility equals the sum of hit Jaccards divided by the full stream length.

This test isolates cache logic from model and dataset complexity.

---

## Expected Output Files

A typical run should produce:

```text
outputs/
└── semantic_cache_dbpedia/
    ├── dbpedia_stream_seed_42.csv
    ├── dbpedia_top1_cache_grid.csv
    ├── embedding_cache/
    │   ├── SBERT-MiniLM_*.npz
    │   ├── SBERT-MPNet_*.npz
    │   └── Qwen3-Embedding-0.6B-zs_*.npz
    ├── traces/
    │   └── *.csv
    ├── dbpedia_utility_K50.png
    ├── dbpedia_utility_K100.png
    └── dbpedia_utility_K200.png
```

The summary CSV should contain at least:

```text
model
capacity
theta
stream_length
hits
misses
hit_rate
quality
utility
reliable_hit_precision_at_0.8
exact_hit_precision
wrong_hit_rate_per_query
catastrophic_hit_rate_per_query
```

---

## Recommended Summary Table

The main experiment table should include:

| Model | Cache capacity | Threshold | Hit rate | Quality | Utility |
|---|---:|---:|---:|---:|---:|

Additional reliability columns may be shown in a second table.

No model should be ranked using hit rate alone.

A high hit rate can indicate an overly permissive embedding space rather than a useful cache.

---

## Interpretation Guidelines

The three primary metrics answer different questions.

### Hit rate

How frequently does the cache avoid execution?

### Quality

When the cache returns a stored result, how closely does that result match the true answer set?

### Utility

Across the entire query stream, how much correct answer-set overlap is served from the cache?

A model may have:

- high hit rate and low quality;
- low hit rate and high quality;
- moderate values for both;
- similar utility to another model through a very different trade-off.

Therefore, utility should always be reported together with hit rate and quality.

---

## Scope of the Top-1 Experiment

This experiment evaluates the answer set actually returned by a deployable top-1 cache policy.

It does not search lower-ranked candidates after observing that the top-1 result has poor Jaccard, because the incoming query's true answer set is unavailable during real deployment.

Top-2 and top-3 analyses may be added later as offline diagnostics, but they should be reported separately from actual cache utility.

Possible future diagnostics include:

- semantic success at \(k\);
- best answer-set Jaccard among the top \(k\);
- rank of the first cached candidate with Jaccard at least \(0.8\);
- top-1 ranking failures where a reliable candidate exists at rank 2 or 3;
- learned reranking among the top cached candidates.

These diagnostics do not change the primary top-1 cache result.

---

## Reproducibility Requirements

Every experiment should record:

```text
dataset
graph snapshot
test split file
stream seed
final stream manifest
model identifier
model revision
prompt configuration
tokenizer configuration
embedding dimension
normalization method
cache capacities
similarity thresholds
answer canonicalization
empty-set convention
tie-breaking policy
software versions
hardware
```

The final stream manifest and summary CSV should be stored with the experiment artifacts.

For stateful FIFO caching, changing stream order can change the cache state and the final metrics. The exact stream order is therefore part of the experimental definition.

---

## Suggested Extensions

Future versions may add:

- SparqlCE;
- fine-tuned answer-supervised baselines;
- validation-selected model-specific thresholds;
- equal-hit-rate operating points;
- multiple stream seeds;
- confidence intervals;
- Wikidata evaluation;
- top-\(k\) candidate diagnostics;
- semantic reranking;
- an oracle cache ceiling;
- cache latency and memory measurements;
- workload-drift experiments;
- graph-snapshot drift analysis.

These extensions should preserve the top-1 FIFO experiment as the main comparable baseline.
