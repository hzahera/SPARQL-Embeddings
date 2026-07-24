# RQ1 — Wikidata DBpedia-matched split: results

Test set: `wikidata_testset_RQ1_matched.json` — 462 anchors, positives
(jaccard>0.8, mean 7.94, cap 20), 20 negatives/anchor; size + thresholds mirror
`dbpedia_testset_RQ1.json`. Corpus 1,968. Reproduce:

```bash
python scripts/eval_rq1_stratified.py --split datasets/wikidata_testset_RQ1_matched.json \
  --st intfloat/e5-base-v2 --st sentence-transformers/all-mpnet-base-v2 \
  --st sentence-transformers/all-MiniLM-L6-v2 --st microsoft/codebert-base \
  --st microsoft/graphcodebert-base --st razent/spbert-mlm-wso-base \
  --st checkpoints/e5_square_wikidata/best_model \
  --code-llm Qwen/Qwen2.5-Coder-0.5B \
  --sparqlce checkpoints_b128/wikidata/seed42_full/best_model.pt
```

Paper table: `tab-RQ1-Wikidata-matched.tex`. Best per column **bold** (all SparqlCE).

| Model | MRR@3 | MRR@5 | MRR@10 | Hits@3 | Hits@5 | Hits@10 | nDCG@3 | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **SparqlCE** | **0.521** | **0.538** | **0.551** | **0.580** | **0.656** | **0.745** | **0.490** | **0.508** | **0.529** |
| TF-IDF | 0.427 | 0.435 | 0.442 | 0.468 | 0.504 | 0.552 | 0.411 | 0.418 | 0.422 |
| BM25 | 0.418 | 0.427 | 0.433 | 0.452 | 0.489 | 0.537 | 0.407 | 0.417 | 0.423 |
| E5-Finetuned | 0.414 | 0.421 | 0.431 | 0.478 | 0.513 | 0.584 | 0.380 | 0.379 | 0.388 |
| E5-base-v2 | 0.396 | 0.411 | 0.421 | 0.446 | 0.506 | 0.582 | 0.357 | 0.366 | 0.374 |
| SPBERT | 0.376 | 0.388 | 0.395 | 0.420 | 0.472 | 0.528 | 0.347 | 0.351 | 0.353 |
| GraphCodeBERT | 0.359 | 0.366 | 0.377 | 0.420 | 0.452 | 0.532 | 0.331 | 0.335 | 0.347 |
| SBERT/MiniLM | 0.340 | 0.353 | 0.365 | 0.377 | 0.433 | 0.530 | 0.307 | 0.311 | 0.325 |
| SBERT/MPNet | 0.338 | 0.347 | 0.359 | 0.396 | 0.437 | 0.526 | 0.306 | 0.311 | 0.320 |
| Qwen2.5-Coder-0.5B | 0.329 | 0.342 | 0.351 | 0.379 | 0.435 | 0.500 | 0.289 | 0.297 | 0.303 |
| CodeBERT | 0.324 | 0.333 | 0.342 | 0.366 | 0.407 | 0.476 | 0.292 | 0.296 | 0.301 |

## Reading the result

- **SparqlCE wins every metric at every cutoff — clean sweep** (bolds all 9
  columns). MRR@10 0.551, Hits@10 0.745, nDCG@10 0.529 — margins of +0.11 / +0.16 /
  +0.11 over the best baseline.
- **Cross-KG consistency:** SparqlCE MRR@10 = 0.551 here vs. 0.544 on the matched
  DBpedia table — near-identical, both decisive wins on structurally identical test
  sets. This is the apples-to-apples cross-KG evidence.
- Lexical methods (TF-IDF/BM25) are the strongest baselines here (2nd/3rd), because
  the natural positive distribution contains many lexically-similar pairs and the
  DBpedia-style shared negatives are weak — yet SparqlCE still leads by a wide
  margin.

## Relation to the other Wikidata sets
- `matched` (this) — DBpedia-matched natural distribution: **SparqlCE sweeps**.
- `adv500` — cos<0.3 adversarial: **SparqlCE sweeps** (larger relative gap).
- `rand500` — random, 10 hard negatives: SparqlCE wins Hits@10/nDCG@10; code-LLMs
  win MRR (hard negatives + multi-positive change the ranking).
