# RQ1 results — retrieval

**Stale checkpoint, superseded for the SparqlCE row.** This file's SparqlCE
numbers (MRR@10 0.544) are from an older checkpoint,
`final_model_mrr_0.4484.pt`. The paper's reported DBpedia RQ1 table
(`tables/tab-RQ1-DBpedia.tex` in the paper repo) uses a newer checkpoint,
`checkpoints/sparqlce_dbpedia/best_model.pt` (MRR@3/5/10 ≈ 0.73/0.74/0.74),
the same one used for the caching experiment
(`results/caching/README.md`). The baseline rows below (TF-IDF, BM25, every
zero-shot encoder) are checkpoint-independent and still valid — only the
**SparqlCE row is stale**. Re-run
`notebooks/evaluation-RQ1/DBpedia/06_eval_sparqlce_dbpedia.ipynb` against
the current checkpoint if you need this file's SparqlCE row updated rather
than relying on the paper table.

Generated from `notebooks/evaluation-RQ1/{DBpedia,Wikidata}/06_eval_sparqlce_*.ipynb`
on the finalized, leak-free test sets (see `datasets/README.md`). Zero-shot
baselines only — no fine-tuned model included in these runs (see the note at
the bottom for how the fine-tuned E5 baseline compares separately).

Test set: `dbpedia_testset_RQ1.json` — 462 anchors / 1841 corpus.
Checkpoint: `final_model_mrr_0.4484.pt` (see stale-checkpoint note above).

## Aggregate (whole test set, all 462 anchors)

| Model | MRR@3 | MRR@5 | MRR@10 | MRR@20 |
|---|---|---|---|---|
| **SparqlCE** | **0.517** | **0.535** | **0.544** | **0.550** |
| SBERT/MPNet | 0.372 | 0.391 | 0.398 | 0.402 |
| E5-base-v2 (zero-shot) | 0.383 | 0.391 | 0.400 | 0.406 |
| SBERT/MiniLM | 0.351 | 0.360 | 0.367 | 0.374 |
| GraphCodeBERT (zero-shot) | 0.308 | 0.317 | 0.324 | 0.329 |
| CodeBERT (zero-shot) | 0.298 | 0.311 | 0.320 | 0.325 |
| SPBERT (zero-shot) | 0.282 | 0.293 | 0.307 | 0.312 |
| Qwen2.5-Coder-0.5B (zero-shot) | 0.248 | 0.259 | 0.272 | 0.278 |
| TF-IDF | 0.240 | 0.253 | 0.260 | 0.265 |
| BM25 | 0.182 | 0.187 | 0.197 | 0.200 |

## Stratified by anchor–positive lexical similarity (TF-IDF cosine, cumulative)

Cumulative thresholds `cos(anchor, positive) < t`, nested from 0 (i.e. `<0.5`
includes everything in `<0.3` plus the 0.3–0.5 slice). `mean_cos`/`median_cos`
report the actual observed cosine values within each slice — never rely on
the threshold label alone.

**MRR@10** by bin:

| Model | Aggregate | <0.3 (n=114) | <0.5 (n=187) | <0.8 (n=376) |
|---|---|---|---|---|
| **SparqlCE** | **0.544** | **0.168** | **0.452** | **0.522** |
| SBERT/MPNet | 0.398 | 0.057 | 0.165 | 0.279 |
| E5-base-v2 (zero-shot) | 0.400 | 0.050 | 0.155 | 0.262 |
| SBERT/MiniLM | 0.367 | 0.067 | 0.165 | 0.262 |
| SPBERT (zero-shot) | 0.307 | 0.041 | 0.150 | 0.259 |
| GraphCodeBERT (zero-shot) | 0.324 | 0.044 | 0.180 | 0.253 |
| CodeBERT (zero-shot) | 0.320 | 0.044 | 0.163 | 0.238 |
| Qwen2.5-Coder-0.5B (zero-shot) | 0.272 | 0.048 | 0.127 | 0.231 |
| TF-IDF | 0.260 | 0.007 | 0.060 | 0.138 |
| BM25 | 0.197 | 0.018 | 0.131 | 0.130 |

`mean_cos` / `median_cos` per bin (SparqlCE run; identical corpus across
models, so these are the same for every model's stratification):

| bin | mean_cos | median_cos | anchors |
|---|---|---|---|
| <0.3 | 0.1847 | 0.1953 | 114 |
| <0.5 | 0.2787 | 0.2921 | 187 |
| <0.8 | 0.4967 | 0.5867 | 376 |

## Reading

- **SparqlCE wins every slice**, and the margin is largest exactly where the
  paper's thesis needs it: ~2.5x the best baseline at `<0.3` (0.168 vs 0.067)
  narrowing to ~1.9x at `<0.8` (0.522 vs 0.279) as more moderately-similar
  positives are folded in. That shape — biggest advantage in the truly
  adversarial regime, still ahead but by less in the easier regime — is the
  honest signature of a result-aware method, not an artifact of test-set
  construction.
- **The `<0.3, <0.5, <0.8` thresholds never explicitly break out the
  remaining ~86 anchors** (462 − 376) with cos ≥ 0.8 (near-duplicate
  lexical positives). We know from an earlier exclusive-bin run that
  baselines actually beat SparqlCE there (E5-zero-shot hit ~0.98 MRR on
  that slice) — that's *folded into* the aggregate column (which is why
  SparqlCE's aggregate lead, 0.544 vs 0.400, looks smaller than its lead
  in the hard slices) but is not its own row here. Add it back explicitly
  if a reviewer needs to see the "where we lose" case spelled out rather
  than inferred from the aggregate gap.
- **Zero-shot only.** A separately-run hard-adversarial check (see
  `datasets/README.md` → "RQ1 hard adversarial sets" and
  `notebooks/evaluation-RQ1/evaluate_hard_adversarial.py`) found that a
  **fine-tuned** E5-base-v2 baseline (full fine-tune, capacity-matched,
  generic contrastive loss — not SparqlCE's architecture) scores *higher*
  than SparqlCE on the sharpest 87-row hard-adversarial slice (0.874 vs
  0.736 ranking-accuracy) and on the full 500-row extension (0.966 vs
  0.886). So: SparqlCE beats every zero-shot baseline decisively (this
  file); a plain fine-tuned baseline is a genuinely tough competitor
  (separate result) — both need to be in the paper. The ablations listed
  in `PAPER_PLAN.md` (−graph, −calibration, −hard-negatives) are the open
  question of whether SparqlCE's specific architecture adds anything over
  generic fine-tuning.

## Pending

- **Wikidata RQ1** — not yet run with the cumulative `<0.3/<0.5/<0.8`
  stratification. Run `notebooks/evaluation-RQ1/Wikidata/06_eval_sparqlce_wikidata.ipynb`
  and this file will be updated with a matching Wikidata section.
