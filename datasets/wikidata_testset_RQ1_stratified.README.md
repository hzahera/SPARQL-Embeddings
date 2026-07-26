# Wikidata RQ1 — stratified (500 natural anchors, reported per cos-bin)

`wikidata_testset_RQ1_stratified.json` — built by
[`scripts/build_stratified_wikidata.py`](../scripts/build_stratified_wikidata.py)
from the held-out Wikidata **test** split.

## What it is and why

A single 500-anchor set spanning the natural lexical-similarity range, reported
**per cos-bin** (`<0.3 / 0.3-0.5 / 0.5-0.8 / >=0.8`) exactly like the DBpedia
stratified table. The aggregate row is a realistic retrieval number; the `<0.3`
stratum isolates the regime with no surface signal, where only a result-aware /
structural model can retrieve. Nothing is hand-picked for difficulty and nothing
is padded with filler — the bins fall out of the natural distribution, so there
is no cherry-pick or dilution artifact (unlike "hard anchors + random padding").

| property | value |
|---|---|
| anchors | 500 (held-out test only, 0 leakage) |
| positives/anchor | min 1, mean 5.88, max 20 (2,942 total) |
| positive answer_jaccard | all > 0.8 |
| negatives/anchor | 20 (shipped, jaccard 0) |
| corpus | 2,194 unique queries |
| lexical bins | computed at eval time (see below) |

## Bins are computed at eval time (not stored)

Binning uses the mean **label-expanded** anchor→positive TF-IDF cosine — the same
expanded text the text baselines read — so it needs the label map, which lives on
the eval host. `eval_rq1_stratified.py --stratify` computes the bins on the fly;
the split file therefore carries no `lexical_bin` field and needs no label map at
build time. (Local raw-text approximation: <0.3≈144, 0.3-0.5≈132, 0.5-0.8≈77,
>=0.8≈147; expanded-text binning shifts some anchors upward.)

## Construction (leak-free, deterministic)

* Source: held-out Wikidata `test_idx` only (no train/val → 0 leakage).
* Anchor kept iff it has ≥ 1 positive with `answer_jaccard > 0.8`.
* Positives: `answer_jaccard > 0.8`, capped at 20 (strongest overlap first).
* Negatives: the shipped `jaccard = 0` distractors, up to 20.
* 500 anchors sampled deterministically (seed 42) from the 1,063 qualifying.

## How to reproduce / evaluate

```bash
python scripts/build_stratified_wikidata.py           # torch-free, no label map needed

python scripts/eval_rq1_stratified.py \
  --split datasets/wikidata_testset_RQ1_stratified.json --stratify \
  --st intfloat/e5-base-v2 --st sentence-transformers/all-mpnet-base-v2 \
  --st sentence-transformers/all-MiniLM-L6-v2 --st microsoft/codebert-base \
  --st microsoft/graphcodebert-base --st razent/spbert-mlm-wso-base \
  --st checkpoints/e5_square_wikidata/best_model \
  --code-llm Qwen/Qwen2.5-Coder-0.5B \
  --sparqlce checkpoints_b128/wikidata/seed42_full/best_model.pt
```

`--stratify` prints a per-bin block (plus the `ALL` aggregate) for every model.
All *text* baselines receive the same Wikidata label-expanded query text SparqlCE
resolves internally (`expand_wikidata_labels`), for a fair comparison.
