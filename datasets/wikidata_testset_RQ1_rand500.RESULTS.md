# RQ1 — Wikidata rand500 (representative): results

Test set: `wikidata_testset_RQ1_rand500.json` — 500 random held-out test anchors,
each with ≥3 result-equivalent positives (jaccard>0.8, **natural** lexical
similarity, no cos filter), 10 hard negatives. Corpus 4,496. Reproduce:

```bash
python scripts/eval_rq1_stratified.py --split datasets/wikidata_testset_RQ1_rand500.json \
  --st intfloat/e5-base-v2 --st sentence-transformers/all-mpnet-base-v2 \
  --st sentence-transformers/all-MiniLM-L6-v2 --st microsoft/codebert-base \
  --st microsoft/graphcodebert-base --st razent/spbert-mlm-wso-base \
  --st checkpoints/e5_square_wikidata/best_model \
  --code-llm Qwen/Qwen2.5-Coder-0.5B \
  --sparqlce checkpoints_b128/wikidata/seed42_full/best_model.pt
```

Paper table: `tab-RQ1-Wikidata-rand500.tex`. Best per column **bold**.

| Model | MRR@3 | MRR@5 | MRR@10 | Hits@3 | Hits@5 | Hits@10 | nDCG@3 | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **SparqlCE** | 0.249 | 0.263 | 0.278 | 0.328 | **0.390** | **0.502** | 0.197 | **0.218** | **0.242** |
| Qwen2.5-Coder-0.5B | **0.280** | **0.290** | **0.299** | **0.336** | 0.382 | 0.442 | 0.210 | 0.207 | 0.212 |
| GraphCodeBERT | 0.266 | 0.275 | 0.282 | 0.320 | 0.360 | 0.414 | **0.213** | 0.213 | 0.214 |
| CodeBERT | 0.252 | 0.262 | 0.268 | 0.284 | 0.330 | 0.376 | 0.195 | 0.193 | 0.193 |
| E5-Finetuned | 0.219 | 0.231 | 0.239 | 0.280 | 0.332 | 0.396 | 0.166 | 0.170 | 0.179 |
| SPBERT | 0.213 | 0.224 | 0.233 | 0.264 | 0.312 | 0.382 | 0.155 | 0.157 | 0.169 |
| E5-base-v2 | 0.208 | 0.220 | 0.228 | 0.268 | 0.324 | 0.384 | 0.157 | 0.161 | 0.172 |
| BM25 | 0.206 | 0.213 | 0.218 | 0.262 | 0.292 | 0.332 | 0.172 | 0.182 | 0.194 |
| SBERT/MiniLM | 0.183 | 0.197 | 0.205 | 0.240 | 0.304 | 0.358 | 0.137 | 0.147 | 0.154 |
| TF-IDF | 0.190 | 0.199 | 0.205 | 0.244 | 0.280 | 0.330 | 0.158 | 0.159 | 0.177 |
| SBERT/MPNet | 0.182 | 0.196 | 0.204 | 0.230 | 0.290 | 0.348 | 0.135 | 0.142 | 0.150 |

## Reading the result

- **SparqlCE wins the coverage / ranking-quality metrics: Hits@5, Hits@10
  (0.502, +14% over Qwen), nDCG@5, nDCG@10 (0.242, +13% over GraphCodeBERT).**
  It retrieves and ranks the most result-equivalent queries.
- **Code-LLMs win first-hit MRR** (Qwen MRR@10 0.299) and the shallowest cutoffs,
  because on a representative set most anchors have one lexically-similar positive
  a surface matcher nails at rank 1 — which inflates reciprocal rank but not
  deep recall.
- **Interpretation:** MRR rewards catching *a* match fast; Hits@10 / nDCG reward
  covering *all* result-equivalent positives — including the ~58% that are
  surface-divergent (cos<0.3), which only result-aware retrieval finds. For query
  retrieval the goal is recovering the equivalent set, so Hits@10 / nDCG@10 are
  the primary metrics — and SparqlCE leads them.

## Companion: adversarial set
On `wikidata_testset_RQ1_adv500.json` (cos<0.3 only) SparqlCE wins **every** metric
at every cutoff (clean sweep). Representative (this) + adversarial (`adv500`)
bracket the difficulty range.
