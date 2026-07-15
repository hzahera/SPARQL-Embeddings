# Multi-seed variance and significance (RQ1)

Cross-seed variance and paired significance testing for the RQ1 retrieval
results, computed with [`scripts/compute_significance.py`](../../scripts/compute_significance.py).

**Corrected 2026-07-16.** The fine-tuned-Qwen3 baseline used for the paired
significance test (`{dbpedia,wikidata}_baseline_qwen3ft.json`) was originally
built with `scripts/eval_rq1_export.py --model-type sentence-transformer`,
which loads a bare LoRA adapter directory via `SentenceTransformer(path)`.
That directory has no `modules.json` (only `adapter_config.json` + adapter
weights), so sentence-transformers silently falls back to
`_load_auto_model()`, which always builds a fresh **mean-pooling** wrapper —
not the **last-token pooling** this model was actually trained and evaluated
with everywhere else. No error is raised. Rerun with the corrected
`--model-type peft-qwen3` (see
[`notebooks/evaluation-RQ1/{DBpedia,Wikidata}/09_fix_qwen3_finetuned_baseline_*.ipynb`](../../notebooks/evaluation-RQ1/)),
DBpedia's baseline MRR@10 moved 0.098 → 0.1093 (now matching the correctly-loaded
0.1091 in `tables/tab-RQ1-DBpedia.tex`) and Wikidata's moved 0.1193 → 0.1939
(matching 0.1927 in `tables/tab-RQ1-Wikidata.tex`) — a much larger correction
on Wikidata, meaning mean-pooling degraded that baseline far more than
DBpedia's. The tables below reflect the corrected numbers; SparqlCE's own
cross-seed stats are unchanged (that side of the comparison was never
affected — it was always loaded via `SPARQLEmbedder`, not
`SentenceTransformer`).

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
  Generate/regenerate it with
  [`notebooks/evaluation-RQ1/{DBpedia,Wikidata}/09_fix_qwen3_finetuned_baseline_*.ipynb`](../../notebooks/evaluation-RQ1/)
  (equivalently, `eval_rq1_export.py --model-type peft-qwen3`) — **not**
  `--model-type sentence-transformer`, which silently mean-pools a bare LoRA
  adapter directory instead of using the last-token pooling this model was
  trained with (see the correction note above).
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

Corrected fine-tuned-Qwen3 baseline: MRR@10 = 0.1093 (was 0.098 with the
mean-pooling bug).

| Metric | SparqlCE mean ± std (5 seeds) | mean diff vs. baseline | p-value | sig. (p<0.05) |
|---|---:|---:|---:|:---:|
| MRR@3   | 0.2220 ± 0.0376 | +0.1360 | 3.58e-10 | yes |
| MRR@5   | 0.2419 ± 0.0367 | +0.1508 | 3.90e-13 | yes |
| MRR@10  | 0.2628 ± 0.0355 | +0.1550 | 7.17e-17 | yes |
| MRR@20  | 0.2751 ± 0.0352 | +0.1533 | 6.61e-18 | yes |
| nDCG@3  | 0.1568 ± 0.0247 | +0.0928 | 4.62e-11 | yes |
| nDCG@5  | 0.1651 ± 0.0228 | +0.0985 | 6.10e-14 | yes |
| nDCG@10 | 0.1879 ± 0.0218 | +0.1075 | 6.07e-19 | yes |
| nDCG@20 | 0.2414 ± 0.0272 | +0.1268 | 1.06e-21 | yes |

### Wikidata (representative seed: **42**, n=186 anchors)

Corrected fine-tuned-Qwen3 baseline: MRR@10 = 0.1939 (was 0.1193 with the
mean-pooling bug — a much larger correction than DBpedia's; mean-pooling hurt
this baseline substantially more).

| Metric | SparqlCE mean ± std (5 seeds) | mean diff vs. baseline | p-value | sig. (p<0.05) |
|---|---:|---:|---:|:---:|
| MRR@3   | 0.2409 ± 0.0270 | +0.0780 | 6.08e-04 | yes |
| MRR@5   | 0.2709 ± 0.0212 | +0.1003 | 3.88e-05 | yes |
| MRR@10  | 0.2871 ± 0.0215 | +0.0931 | 7.21e-06 | yes |
| MRR@20  | 0.2929 ± 0.0216 | +0.0896 | 8.31e-06 | yes |
| nDCG@3  | 0.1902 ± 0.0261 | +0.0513 | 3.13e-03 | yes |
| nDCG@5  | 0.2044 ± 0.0132 | +0.0458 | 5.25e-03 | yes |
| nDCG@10 | 0.2253 ± 0.0095 | +0.0438 | 1.14e-03 | yes |
| nDCG@20 | 0.2722 ± 0.0136 | +0.0467 | 9.30e-04 | yes |

### Reading these numbers

- All 16 metric/graph comparisons remain significant after the fix, but the
  margin over the fine-tuned baseline is now visibly narrower on Wikidata
  (mean diff ~0.05-0.10, p down to 6e-4) than on DBpedia (mean diff
  ~0.09-0.16, p down to 6e-19) — the corrected, stronger Wikidata baseline
  closes much of the gap that the mean-pooling bug had artificially widened.
  This is consistent with, and reinforces, the narrower Wikidata margin
  already noted for the aggregate RQ1/RQ2 gaps in the paper's Limitations
  section ("Unattributed margin over the fine-tuned baseline").
- Std is small relative to mean everywhere (roughly 8-17%) — unaffected by
  the fix, since SparqlCE's own eval never used the buggy loading path.
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
