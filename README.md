# Results-Aware SPARQL Embeddings (SQuARE)

Code and data for the paper **"Contrastive Learning for Results-Aware SPARQL Embeddings"**.

Two SPARQL queries can look almost the same and return different answers, or look different and return the same answers. We execute a query log **once, offline**, label query pairs by the **Jaccard overlap of their answer sets**, and fine-tune pretrained encoders with a contrastive loss on these labels. At inference, the encoder needs **only the query text**.

## ▶ Interactive demo

![Animated preview of the interactive demo](docs/media/demo_preview.gif)

Runs in the browser, nothing to install. It shows the problem on real test queries, the pipeline step by step (with live toy training), and the effect on query retrieval and result caching. The page is one self-contained file, [`docs/index.html`](docs/index.html) (sources in [`demo/`](demo/)), served as a static website.

## Key results

Ten pretrained encoders (text, code, general LM, SPARQL), before and after results-aware fine-tuning. Mean over encoders; fine-tuned = mean of five seeds.

| | DBpedia | Wikidata |
|---|---|---|
| **Query retrieval**, Hits@1 (complete labels): TF-IDF | 0.58 | 0.37 |
| zero-shot encoders | 0.49 | 0.30 |
| **results-aware fine-tuned encoders** | **0.70** | **0.43** |
| Encoders that improve | 10 / 10 | 10 / 10 |
| Control: 3 encoders trained on **syntax** labels (vs. answer labels) | 0.52 (vs. 0.72) | 0.36 (vs. 0.43) |
| **Result caching**, hit accuracy at θ = 0.8: zero-shot → fine-tuned | 0.15–0.33 → **0.59–0.79** | 0.01–0.10 → **0.21–0.37** |

A fixed similarity threshold is still not enough for safe reuse (see the paper, RQ3): ranking quality and calibration are distinct problems.

## The SQuARE resource

| | DBpedia | Wikidata |
|---|---|---|
| Unique queries (train / valid / test) | 3,781 / 473 / 473 | 7,743 / 968 / 968 |
| Positive pairs (J ≥ 0.8) | 39,430 | 37,208 |
| Negative pairs (J ≤ 0.1) | 93,480 | 168,360 |

Splits are built from connected components of positive pairs, so no query, positive pair or answer set is shared across splits. Files: `datasets/{dbpedia,wikidata}-splits/` (`*-train/valid/test.json` for training and retrieval, `*-test-rq3.json` with complete pair labels for caching).

```json
{"anchor_query": "SELECT ...",
 "positive_queries": [{"query": "SELECT ...", "answer_jaccard": 1.0}],
 "negative_queries": [{"query": "SELECT ...", "answer_jaccard": 0.04}]}
```

## Quick start

```bash
pip install -e .
python scripts/validate_data.py --dataset dbpedia          # sanity checks on the splits

# one model
python scripts/run_experiment.py --models e5-base-v2 --datasets dbpedia --seeds 42 43 44 45 46

# full benchmark (10 encoders x 2 graphs x 5 seeds), then the paper tables
python scripts/run_experiment.py --models all --datasets dbpedia wikidata --seeds 42 43 44 45 46
python scripts/aggregate_results.py && python scripts/generate_tables.py
```

Each run evaluates zero-shot, fine-tunes, selects the checkpoint on validation nDCG@10, and evaluates retrieval, alignment and caching on test. Runs resume; use `--force` to recompute. Gated models (Gemma) need `huggingface-cli login`.

## Repository layout

```text
src/result_aware/   training (loss.py, mining.py, trainer.py), evaluation, caching simulation
configs/            experiment.yaml (all hyperparameters) and one YAML per encoder
scripts/            run_experiment.py, aggregate_results.py, generate_tables.py
  baselines/        non-neural baselines (TF-IDF, BM25, algebra similarity, ...)
  syntax_control/   control: same training with syntax-based labels
  analysis/         thresholds, timing, paired bootstrap, per-anchor scores
  audit/            split leakage audit
datasets/           SQuARE splits and statistics
demo/, docs/        interactive demo (sources and single-file build)
tests/              unit tests (pytest -q, no model downloads)
```

## Method in brief

InfoNCE (temperature 0.05) with in-batch candidates, 4 execution-verified explicit negatives per anchor and known positives masked. From epoch 2, every two epochs, the current model's 8 hardest **labeled** negatives are mined (75% hard / 25% random). AdamW, lr 1e-5, 20 epochs, batch 4 x 8 accumulation. Validation and test queries are never seen by the miner, and the cache threshold is never tuned on test. To add an encoder: `python scripts/add_model.py --key <name> --hf-id <hf-model> ...`, no code changes.

## License

Code: MIT. Data: CC BY 4.0. Queries derive from LSQ; its attribution requirements apply.
