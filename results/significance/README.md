# Multi-seed variance and significance (RQ1)

Cross-seed variance and paired significance testing for the RQ1 retrieval
results, computed with [`scripts/compute_significance.py`](../../scripts/compute_significance.py).

## What this answers

1. **Cross-seed variance for SparqlCE** — mean ± std of MRR/nDCG over 5 seeds,
   per knowledge graph. Shows whether the headline RQ1 result depends on a
   single lucky seed.
2. **Paired significance vs. the strongest baseline** — a one-sided Wilcoxon
   signed-rank test on per-anchor MRR/nDCG (SparqlCE vs. baseline on the same
   anchors), using the seed whose score is closest to the cross-seed mean (a
   representative seed rather than the best one).

## Protocol

- **Seeds:** 5 full retrains from scratch per KG (42, 43, 44, 45, 46), same
  architecture/hyperparameters as the main run — only the seed changes.
- **Baseline:** fine-tuned Qwen3, the strongest baseline on both graphs.
- **Test:** one-sided Wilcoxon signed-rank (`scipy.stats.wilcoxon`,
  `alternative="greater"`), paired per-anchor, at the representative seed only.

```bash
python scripts/compute_significance.py \
    --sparqlce-glob "results/significance/dbpedia_seed*_full.json" \
    --baseline results/significance/dbpedia_baseline_qwen3ft.json \
    --out results/significance/dbpedia_full_summary.json
# swap dbpedia -> wikidata for the other graph
```

## Results

### DBpedia (representative seed: **44**, n=337 anchors)

| Metric | SparqlCE mean ± std (5 seeds) | mean diff vs. baseline | p-value | sig. (p<0.05) |
|---|---:|---:|---:|:---:|
| MRR@3   | 0.2220 ± 0.0376 | +0.1429 | 6.80e-10 | yes |
| MRR@5   | 0.2419 ± 0.0367 | +0.1558 | 5.05e-11 | yes |
| MRR@10  | 0.2628 ± 0.0355 | +0.1662 | 4.13e-14 | yes |
| MRR@20  | 0.2751 ± 0.0352 | +0.1718 | 3.04e-17 | yes |
| nDCG@3  | 0.1568 ± 0.0247 | +0.1030 | 8.75e-11 | yes |
| nDCG@5  | 0.1651 ± 0.0228 | +0.1149 | 1.02e-13 | yes |
| nDCG@10 | 0.1879 ± 0.0218 | +0.1344 | 2.21e-20 | yes |
| nDCG@20 | 0.2414 ± 0.0272 | +0.1835 | 1.94e-29 | yes |

### Wikidata (representative seed: **42**, n=186 anchors)

| Metric | SparqlCE mean ± std (5 seeds) | mean diff vs. baseline | p-value | sig. (p<0.05) |
|---|---:|---:|---:|:---:|
| MRR@3   | 0.2409 ± 0.0270 | +0.1362 | 1.45e-08 | yes |
| MRR@5   | 0.2709 ± 0.0212 | +0.1663 | 9.48e-13 | yes |
| MRR@10  | 0.2871 ± 0.0215 | +0.1677 | 2.52e-14 | yes |
| MRR@20  | 0.2929 ± 0.0216 | +0.1647 | 2.57e-15 | yes |
| nDCG@3  | 0.1902 ± 0.0261 | +0.1012 | 2.53e-08 | yes |
| nDCG@5  | 0.2044 ± 0.0132 | +0.1061 | 5.08e-10 | yes |
| nDCG@10 | 0.2253 ± 0.0095 | +0.1305 | 9.46e-16 | yes |
| nDCG@20 | 0.2722 ± 0.0136 | +0.1773 | 3.14e-21 | yes |

### Reading these numbers

- All 16 metric/graph comparisons are significant, most by many orders of
  magnitude past p<0.05 — the RQ1 gain over the strongest baseline is not a
  single-run artifact.
- Std is small relative to mean everywhere (roughly 8-17%).
- Std is noticeably tighter on Wikidata than DBpedia for the higher-k/nDCG
  metrics (e.g. nDCG@10: 0.0095 vs. 0.0218) — consistent with Wikidata's
  near-binary Jaccard distribution making the evaluation pool less noisy.

## Files

- `{dbpedia,wikidata}_seed{42..46}_full.json` — per-seed SparqlCE eval export
  (`aggregate` + `per_anchor` scores), one per seed.
- `{dbpedia,wikidata}_baseline_qwen3ft.json` — fine-tuned Qwen3 eval export,
  same format, used as the paired-test baseline.
- `{dbpedia,wikidata}_full_summary.json` — output of `compute_significance.py`;
  source of the tables above.
