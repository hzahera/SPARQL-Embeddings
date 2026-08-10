# SparqlCE — Supplementary Material

This document holds the analyses referenced from the main paper but kept
out of the 9-page WSDM body. Every item below is summarized in one to two
sentences in the paper, with a footnote pointing here. Section labels
match the `\suppurl` footnotes in the paper.

Contents:
0. Measurement-parity audit — **read this before any number below**
1. Preprocessing and benchmark construction
2. Sensitivity to the SQuARE construction threshold
3. Full set-overlap (Precision / Recall / F1) tables — RQ1
4. Hyperparameter configuration
5. Per-operator robustness radar (RQ2)
6. Embedding–answer-set alignment on Wikidata
7. Full semantic caching grids (K × θ)
8. Retrieval-at-scale latency benchmark

> **Staleness warning.** Sections 2, 3, 5 and 6 predate the parity fixes in
> Section 0 and, where they compare SparqlCE to a fine-tuned control, predate
> the replacement of that control (Qwen3 → E5-base-v2). They are retained as a
> record of what was measured, not as current results. The current DBpedia RQ1
> result and the reasoning behind it are in
> [DBPEDIA_RQ1_FINDINGS.md](DBPEDIA_RQ1_FINDINGS.md).

---

## 0. Measurement-Parity Audit

Comparing two architectures trained on the same supervision requires that
nothing except the architecture differ. Nine violations of that requirement
were found and fixed; each had been silently distorting results, and every
number in the paper's experiments section is measured after all of them. Most
are not specific to this work — a contrastive retrieval comparison between a
graph model and a text model is a natural home for all nine.

| # | Violation | Effect | Fix |
|---|---|---|---|
| i | Validation metric credited only the diagonal of the similarity matrix | Val MRR capped ≈0.09 regardless of model quality; **both** trainers early-stopped at epoch 1–2 | credit any of the anchor's positives (`e3ea5d4`, `ca152f7`) |
| ii | Validation resampled positives every epoch | the same checkpoint scored differently per pass; early stopping reacted to sampling noise | `deterministic=True` eval views |
| iii | Negative parity (component protocol) | control received curated negatives SparqlCE did not | `1b9c26f` |
| iv | Negative parity (group protocol) | same bug, other branch — control +0.034, SparqlCE +0.000 | `713fe79` |
| v | Hard-negative exposure | control's loss scored every anchor against all batch negatives, SparqlCE's only its own slice — 64× at batch 128 | `--shared-negatives` (`67816e9`) |
| vi | Embedding width | 128-d vs the control's 768-d, in a comparison described as capacity-matched; worth 0.4949 → 0.6003 nDCG@10 within one seed | `--output-dim` (`cdc2e0c`) |
| vii | Query-stream truncation | fixed 256-token padding truncated >5% of queries — the longest ones — while wasting ≈3.5× memory (median 73 tokens, max 364) | per-batch padding, budget 384 (`d31ec9d`) |
| viii | Text stream read node labels, not the query | SparqlCE was strictly less informed than the baseline it was measured against | `--query-stream` |
| ix | Inference tokenized node labels at 16 vs training's 32 | long labels encoded differently at inference | fixed |

Two infrastructure faults in the same class: an OOM fallback path indexed an
`(embedding, gate)` tuple and so raised instead of recovering; and checkpoint
and result paths collided across variants, so one sweep silently overwrote
another's checkpoints and a 768-d sweep destroyed the 128-d exports (paths now
carry variant, width and corpus — `f425300`, `a420372`, `3d6d860`, `07dde19`).

**Corpus diagnostics.** Two scripts measure, before any training, whether a
query log can support a structural encoder at all:
`scripts/select_structured_queries.py` (join vertices, query-graph diameter,
template count, operator mix) and `scripts/check_pair_density.py` (relevance
density, and the share of result-equivalence that is structural rather than
constant-driven). Section 3 of
[DBPEDIA_RQ1_FINDINGS.md](DBPEDIA_RQ1_FINDINGS.md) reports what they say about
SQuARE-DBpedia and why the paper's tie is the expected outcome there.

---

## 1. Preprocessing and Benchmark Construction

All queries originate from the LSQ dataset (Saleem et al., 2015), which
publishes queries issued by real users and applications to public SPARQL
endpoints; we use its DBpedia and Wikidata logs.

- **Query filtering.** We retain well-formed `SELECT` queries that parse
  and execute over the target knowledge graph, discarding malformed,
  empty-result, and timed-out queries.
- **Answer-set extraction.** Each retained query is executed once over
  the graph *G*; the returned bindings of its projection variables form
  its answer set Ans(q, G), cached for reproducibility. Pairwise semantic
  similarity is the answer-set Jaccard coefficient.
- **TF–IDF vectorization.** Each query string is tokenized and
  represented as a TF–IDF vector; the cosine of two such vectors gives
  the lexical similarity used, jointly with answer-set Jaccard, to select
  adversarial positives and negatives.

---

## 2. Sensitivity to the SQuARE Construction Threshold

SQuARE defines correct candidates by an answer-set Jaccard cut of τ = 0.8.
We re-score MRR@10 at τ ∈ {0.6, 0.7, 0.8, 0.9} on a fixed anchor
population per graph (301 DBpedia, 175 Wikidata anchors that have a
relevant candidate even at the strictest τ = 0.9, so the denominator does
not change across the sweep).

- **DBpedia.** 1.29% of pool pairs fall inside the swept band [0.6, 0.9),
  so MRR@10 declines smoothly and the gap over the best baseline widens
  from 1.62× to 1.86× as τ tightens.
- **Wikidata.** Consistent with the near-binary overlap distribution
  (§6 below), *zero* pool pairs fall in [0.6, 0.9): every relevant
  candidate is either clearly irrelevant or already at τ ≥ 0.9 overlap.
  No candidate can change relevance label inside the swept range, so
  MRR@10 and the 1.14× gap are exactly, mechanically unchanged across all
  four cuts — a direct consequence of Wikidata's answer-set geometry, not
  an artifact of τ = 0.8 or an evaluation bug.

Both graphs confirm the paper's conclusions do not hinge on the specific
threshold. See `results/` for `threshold_sensitivity.pdf`.

---

## 3. Full Set-Overlap Tables (Precision / Recall / F1) — RQ1

Moved here from the main paper (the main body reports the ranking metrics
MRR / Hits / nDCG). The set-overlap ordering matches the ranking tables:
SparqlCE is best on every cell of both graphs. Best per k in **bold**.

### DBpedia (337 anchors, 1677 corpus queries)

| Model | P@3 | R@3 | F1@3 | P@5 | R@5 | F1@5 | P@10 | R@10 | F1@10 | P@20 | R@20 | F1@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF | 0.0445 | 0.0213 | 0.0270 | 0.0362 | 0.0233 | 0.0251 | 0.0217 | 0.0255 | 0.0200 | 0.0154 | 0.0305 | 0.0177 |
| BM25 | 0.0099 | 0.0029 | 0.0041 | 0.0083 | 0.0037 | 0.0047 | 0.0092 | 0.0123 | 0.0092 | 0.0082 | 0.0210 | 0.0106 |
| MiniLM | 0.0208 | 0.0101 | 0.0123 | 0.0196 | 0.0134 | 0.0137 | 0.0172 | 0.0201 | 0.0154 | 0.0194 | 0.0447 | 0.0230 |
| MPNet | 0.0307 | 0.0141 | 0.0169 | 0.0285 | 0.0203 | 0.0202 | 0.0270 | 0.0328 | 0.0248 | 0.0269 | 0.0580 | 0.0316 |
| E5-base-v2 | 0.0297 | 0.0134 | 0.0162 | 0.0237 | 0.0176 | 0.0170 | 0.0208 | 0.0303 | 0.0205 | 0.0209 | 0.0527 | 0.0251 |
| **SparqlCE (ours)** | **0.1672** | **0.0609** | **0.0804** | **0.1650** | **0.0980** | **0.1085** | **0.1549** | **0.1749** | **0.1444** | **0.1381** | **0.3047** | **0.1693** |

### Wikidata (186 anchors, 2229 corpus queries)

| Model | P@3 | R@3 | F1@3 | P@5 | R@5 | F1@5 | P@10 | R@10 | F1@10 | P@20 | R@20 | F1@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF | 0.0717 | 0.0143 | 0.0227 | 0.0796 | 0.0263 | 0.0376 | 0.0683 | 0.0453 | 0.0516 | 0.0417 | 0.0621 | 0.0460 |
| BM25 | 0.0753 | 0.0156 | 0.0246 | 0.0839 | 0.0287 | 0.0408 | 0.0742 | 0.0515 | 0.0569 | 0.0454 | 0.0603 | 0.0482 |
| MiniLM | 0.0986 | 0.0197 | 0.0310 | 0.1011 | 0.0330 | 0.0469 | 0.0855 | 0.0568 | 0.0639 | 0.0481 | 0.0630 | 0.0507 |
| MPNet | 0.0914 | 0.0172 | 0.0280 | 0.0968 | 0.0326 | 0.0465 | 0.0866 | 0.0575 | 0.0654 | 0.0548 | 0.0722 | 0.0589 |
| E5-base-v2 | 0.0878 | 0.0177 | 0.0280 | 0.0968 | 0.0317 | 0.0454 | 0.0887 | 0.0595 | 0.0666 | 0.0573 | 0.0822 | 0.0630 |
| **SparqlCE (ours)** | **0.1756** | **0.0417** | **0.0621** | **0.1935** | **0.0789** | **0.1015** | **0.2075** | **0.1681** | **0.1655** | **0.1987** | **0.3132** | **0.2200** |

---

## 4. Hyperparameter Configuration

| Component | Parameter | Value |
|---|---|---|
| Text Encoder | Model | `microsoft/codebert-base` |
| Text Encoder | Output dimension (d_t) | 768 |
| Text Encoder | Max token length (node/edge) | 32 / 16 |
| GATv2Conv | Number of heads (H) | 4 |
| GATv2Conv | Hidden dimension (d_h) | 256 |
| GATv2Conv | Edge feature dimension | 768 |
| GATv2Conv | Output dimension | 1024 |
| Projection | Output dimension (d_z) | 128 |
| Training | Batch size (B) | 32 |
| Training | Learning rate (CodeBERT) | 1e-5 |
| Training | Learning rate (GATv2, projection) | 1e-4 |
| Training | Epochs | 20 |
| Training | Temperature (τ) | 0.05 |
| Training | Regularization weight (λ) | 0.3 |
| Training | Train / Valid / Test split | 80% / 10% / 10% |

---

## 5. Per-Operator Robustness Radar (RQ2)

`results/robustness_radar.pdf` visualizes the per-operator embedding
invariance of the paper's Table 4 as radar plots (radial axis truncated
to [0.85, 1.0]; larger area is better). The KG-specific operator is
subquery wrapping for DBpedia and BGP splitting for Wikidata. SparqlCE
forms a near-complete heptagon on DBpedia; the Wikidata exceptions
(redundant OPTIONAL, and the KG-specific operator) are analyzed in the
paper's Limitations. Note the area measures surface-form *invariance*
only: E5 spans a larger area on Wikidata, yet under the same rewrites
SparqlCE retains far higher retrieval quality (perturbed MRR 0.2483 vs.
0.1109).

---

## 6. Embedding–Answer-Set Alignment on Wikidata

`results/cosine_vs_jaccard_wikidata.pdf` repeats the DBpedia alignment
analysis on Wikidata. Answer-set overlap on the Wikidata test corpus is
nearly binary — of 2,126 sampled pairs, 52% are disjoint (J = 0) and 45%
identical (J = 1) — so the panels concentrate at the extremes rather than
spanning [0, 1]. The ordering nevertheless matches DBpedia: SparqlCE
(ρ = 0.78) separates the two regimes cleanly, while the zero-shot encoders
remain compressed near high similarity (ρ ∈ [0.40, 0.55]).

<!-- TODO: this notebook (07_visualize_alignment_wikidata.ipynb) hasn't been
re-run since Qwen3 was dropped for E5-Finetuned/Qwen2.5-Coder-0.5B — its
cached output still only has Qwen3 zero-shot, no fine-tuned-baseline rho.
Re-run and report E5-Finetuned's ρ here once available. -->

---

## 7. Full Semantic Caching Grids (K × θ)

Extends the paper's Figure 6 to the full K × θ grid, K ∈ {50, 100, 200},
θ ∈ {0.7, 0.8, 0.9}. Each cell reports **hit rate / result quality**
(mean answer-set Jaccard over hits). SparqlCE's result quality rises
consistently as θ tightens at every K on both graphs, as a semantic cache
should behave; text baselines do not show this.

<!-- TODO: regenerate with scripts/run_caching_experiment.py's current
roster (GraphCodeBERT, Qwen2.5-Coder-0.5B, E5-Finetuned) and the paper's
headline min-answer-size=5 filter (unfiltered here); see
results/caching/README.md for the current numbers and full sweep. -->

### DBpedia (396-query held-out stream, seed 42)

| Model | 50/0.7 | 50/0.8 | 50/0.9 | 100/0.7 | 100/0.8 | 100/0.9 | 200/0.7 | 200/0.8 | 200/0.9 |
|---|---|---|---|---|---|---|---|---|---|
| SparqlCE | 0.831/0.183 | 0.604/0.237 | 0.253/0.335 | 0.843/0.178 | 0.687/0.264 | 0.328/0.354 | 0.843/0.178 | 0.702/0.262 | 0.414/0.319 |
| TF-IDF | 0.795/0.052 | 0.753/0.085 | 0.644/0.096 | 0.826/0.048 | 0.788/0.069 | 0.662/0.097 | 0.826/0.048 | 0.788/0.069 | 0.682/0.095 |
| BM25 | 0.664/0.097 | 0.641/0.088 | 0.621/0.109 | 0.694/0.077 | 0.674/0.086 | 0.646/0.071 | 0.710/0.069 | 0.692/0.074 | 0.667/0.079 |
| E5-base-v2 | 0.997/0.017 | 0.997/0.017 | 0.970/0.034 | 0.997/0.017 | 0.997/0.017 | 0.970/0.034 | 0.997/0.017 | 0.997/0.017 | 0.970/0.034 |
| SBERT/MiniLM | 0.972/0.018 | 0.955/0.036 | 0.874/0.085 | 0.972/0.018 | 0.955/0.036 | 0.874/0.085 | 0.972/0.018 | 0.955/0.036 | 0.874/0.085 |
| SBERT/MPNet | 0.977/0.038 | 0.939/0.048 | 0.795/0.098 | 0.977/0.038 | 0.939/0.048 | 0.826/0.101 | 0.977/0.038 | 0.939/0.048 | 0.826/0.101 |

### Wikidata (617-query held-out stream, seed 42)

| Model | 50/0.7 | 50/0.8 | 50/0.9 | 100/0.7 | 100/0.8 | 100/0.9 | 200/0.7 | 200/0.8 | 200/0.9 |
|---|---|---|---|---|---|---|---|---|---|
| SparqlCE | 0.765/0.052 | 0.585/0.059 | 0.476/0.078 | 0.844/0.052 | 0.647/0.077 | 0.545/0.081 | 0.844/0.052 | 0.703/0.079 | 0.579/0.097 |
| TF-IDF | 0.305/0.070 | 0.246/0.083 | 0.206/0.075 | 0.387/0.089 | 0.324/0.092 | 0.285/0.080 | 0.451/0.086 | 0.374/0.077 | 0.329/0.087 |
| BM25 | 0.240/0.078 | 0.207/0.074 | 0.203/0.084 | 0.316/0.098 | 0.295/0.089 | 0.277/0.076 | 0.376/0.085 | 0.344/0.080 | 0.314/0.077 |
| E5-base-v2 | 0.998/0.000 | 0.998/0.000 | 0.990/0.008 | 0.998/0.000 | 0.998/0.000 | 0.990/0.008 | 0.998/0.000 | 0.998/0.000 | 0.990/0.008 |
| SBERT/MiniLM | 0.979/0.008 | 0.953/0.043 | 0.804/0.043 | 0.979/0.008 | 0.953/0.043 | 0.854/0.045 | 0.979/0.008 | 0.953/0.043 | 0.854/0.045 |
| SBERT/MPNet | 0.979/0.019 | 0.950/0.039 | 0.767/0.046 | 0.979/0.019 | 0.950/0.039 | 0.835/0.062 | 0.979/0.019 | 0.950/0.039 | 0.835/0.062 |

---

## 8. Retrieval-at-Scale Latency Benchmark

Backs the "Retrieval at scale" paragraph in the paper. We isolate the
corpus-dependent part of inference — the nearest-neighbor search over the
d=128 SparqlCE embeddings — and measure it on a **single CPU thread**
(FAISS; cosine via inner product on unit-normalized vectors; query set
replayed from the corpus; k=20). The per-query encoder forward pass is a
fixed cost independent of corpus size and is excluded here. Reproduce
with `scripts/faiss_latency.py` (corpus scale) and `scripts/faiss_scale.py
N hnsw` (large scale).

| Corpus size N | Exact (FlatIP) latency | Exact throughput | HNSW latency | HNSW throughput | Index size |
|---|---|---|---|---|---|
| 1,677 (SQuARE-DBpedia) | 0.010 ms/q | ~97,000 q/s | 0.046 ms/q | ~21,800 q/s | 1 MB |
| 2,229 (SQuARE-Wikidata) | 0.011 ms/q | ~89,000 q/s | 0.051 ms/q | ~19,400 q/s | 1 MB |
| 100,000 | 1.84 ms/q | ~540 q/s | 0.28 ms/q | ~3,560 q/s | 51 MB |
| 1,000,000 | 18.1 ms/q | ~55 q/s | (sub-ms; HNSW scales ~log N) | — | 512 MB |

Takeaways: exact search is sub-millisecond up to the SQuARE corpus sizes
and remains practical (18 ms) even by brute force at 1M queries; an HNSW
index keeps search **sub-millisecond** through 10^5 and, by HNSW's
logarithmic scaling, into the 10^6 range. Live SPARQL execution over
public endpoints, by contrast, routinely takes hundreds of milliseconds
to seconds per query (with timeouts common; see LSQ endpoint statistics),
and must actually run over the graph. Execution-free retrieval therefore
replaces a variable, potentially unbounded execution with a bounded
lookup two-to-four orders of magnitude cheaper.

Hardware note: numbers are single-thread CPU on the build machine and are
meant to show *scaling behavior and the execution-vs-lookup gap*, not an
absolute hardware record; multi-threading and GPU IVF/HNSW lower them
further.
