# Run Guide — models & experiments (server)

## Model roster (final)

| # | Model | Type | Notes |
|---|-------|------|-------|
| 1 | **SparqlCE** (ours) | trained | CodeBERT + GATv2 graph encoder + calibrated loss |
| 2 | TF-IDF | zero-shot lexical | |
| 3 | BM25 | zero-shot lexical | |
| 4 | SBERT/MiniLM | zero-shot | `sentence-transformers/all-MiniLM-L6-v2` |
| 5 | SBERT/MPNet | zero-shot | `sentence-transformers/all-mpnet-base-v2` |
| 6 | E5-base-v2 | zero-shot | `intfloat/e5-base-v2`, `"query: "` prefix |
| 7 | CodeBERT | zero-shot | `microsoft/codebert-base` (SparqlCE's own backbone) |
| 8 | GraphCodeBERT | zero-shot | `microsoft/graphcodebert-base` |
| 9 | SPBERT | zero-shot | `razent/spbert-mlm-wso-base` (BERT pretrained on SPARQL) |
| 10 | Qwen2.5-Coder-0.5B | zero-shot | generative code LLM used as embedder (last-token); replaced Qwen3-Embedding |
| 11 | **E5-base-v2 (fine-tuned)** | fine-tuned baseline | capacity-matched (~110M), full fine-tune; replaced Qwen3-ft |

Qwen3-Embedding is **dense** (not MoE) and has been dropped in favor of Qwen2.5-Coder per decision.

## Prerequisites (once)

- `git pull` — confirm you're at the latest commit on `main-refactored`.
- `pip install -r requirements.txt`
- Large files not in git — place under `datasets/`: `wikidata-triplets-dataset.json` (~108 MB), `result_sets_wikidata.json` (~6 GB).
- SparqlCE checkpoint at `sparqlce/models/final_model_mrr_*.pt` (produced by Step 1). Update `CHECKPOINT_PATH` in eval notebooks if the filename differs.
- (Optional) re-verify leak-free RQ1/RQ2 test sets: anchor→anchor leak must be 0 for both KGs.

## Step 1 — Train SparqlCE

- `notebooks/training/train_dbpedia.ipynb`
- `notebooks/training/train_wikidata.ipynb`

## Step 2 — Fine-tune the E5 baseline

- `notebooks/finetuning/04_e5_contrastive_finetune_square.ipynb` (DBpedia → `checkpoints/e5_square_dbpedia/best_model`)
- `notebooks/finetuning/05_e5_contrastive_finetune_square_wikidata.ipynb` (Wikidata → `checkpoints/e5_square_wikidata/best_model`)

## Step 3 — RQ1 (retrieval)

- `notebooks/evaluation-RQ1/DBpedia/06_eval_sparqlce_dbpedia.ipynb`
- `notebooks/evaluation-RQ1/Wikidata/06_eval_sparqlce_wikidata.ipynb`
- `notebooks/evaluation-RQ1/{DBpedia,Wikidata}/07_visualize_alignment_*.ipynb` (Fig. 3 ρ)
- (optional) `notebooks/evaluation-RQ1/threshold_sensitivity_{dbpedia,wikidata}.ipynb`

### Step 3b — RQ1 hard-adversarial sets (motivating-example evaluation)

DBpedia — already built and committed, `datasets/dbpedia_testset_RQ1_hard.json`
(472 unique anchors, padded to 500 rows; 87 meet the full strict bar — see
`datasets/README.md`). Re-run only if you change thresholds:

```bash
python scripts/build_hard_adversarial_rq1.py \
  --kg dbpedia --triplets datasets/dbpedia-triplets-dataset.json \
  --splits datasets/splits_dbpedia.json \
  --result-sets datasets/result_sets_dbpedia.json \
  --out datasets/dbpedia_testset_RQ1_hard.json --target-n 500
```

Wikidata — **must run on the cluster** (result_sets_wikidata.json is 6.4GB,
individual result sets run to millions of rows; not tractable in a
memory-constrained environment). Single streaming pass, so runtime is
dominated by disk I/O reading the whole 6.4GB file once — expect it to take a
while:

```bash
python scripts/build_hard_adversarial_rq1_wikidata.py \
  --triplets datasets/wikidata-triplets-dataset.json \
  --splits datasets/splits_wikidata.json \
  --result-sets datasets/result_sets_wikidata.json \
  --out datasets/wikidata_testset_RQ1_hard.json \
  --target-n 500 --top-k-candidates 30
```

Send me the printed summary line (`N rows from M unique anchors ... K/N meet
the full strict bar`) once it finishes — I'll fold the real numbers into the
paper text instead of guessing.

## Step 4 — RQ2 (robustness)

- `notebooks/evaluation-RQ2/DBpedia/04_eval_sparql_RQ2_dbpedia.ipynb`
- `notebooks/evaluation-RQ2/DBpedia/04b_eval_e5_finetuned_RQ2_dbpedia.ipynb`
- `notebooks/evaluation-RQ2/Wikidata/04_eval_sparql_RQ2_wikidata.ipynb`
- `notebooks/evaluation-RQ2/Wikidata/04b_eval_e5_finetuned_RQ2_wikidata.ipynb`
- `notebooks/evaluation-RQ2/{DBpedia,Wikidata}/08_visualize_robustness_*.ipynb`

## Step 5 — Downstream application

- `notebooks/caching/caching_simulation_dbpedia.ipynb` — ✅ updated with new roster (RUN THIS FIRST to verify)
- `notebooks/caching/caching_simulation_wikidata.ipynb` — ⚠ NOT YET updated (pending)
- `notebooks/compaction/query_log_compaction_dbpedia.ipynb` — ⚠ NOT YET updated (pending)

## Step 6 — Significance testing (optional, for paper's significance tables)

- Train 5 seeds/KG: `scripts/run_seed_sweep.sh` or `scripts/train_sparqlce.py --kg {dbpedia,wikidata} --seed {42..46}`
- Export per seed: `scripts/eval_rq1_export.py --kg <kg> --model-type sparqlce --checkpoint <ckpt> --seed N --out results/significance/<kg>_seedN_full.json`
- Export fine-tuned baseline: ⚠ the `--model-type peft-qwen3` path still targets the old Qwen3-ft adapter; needs updating for the E5-ft full model before this row is valid.
- `scripts/compute_significance.py --sparqlce-glob "results/significance/<kg>_seed*_full.json" --baseline <baseline.json> --out results/significance/<kg>_full_summary.json`

## Known gaps to close after runs

- Caching-Wikidata + compaction notebooks: apply the same baseline roster edits (blocked on verifying caching-DBpedia first).
- `scripts/eval_rq1_export.py`: replace the `peft-qwen3` significance baseline path with the E5-ft full model.
- `SUPPLEMENTARY.md` tables + paper rows: still labeled Qwen3; regenerate/rename with the E5-ft and Qwen2.5-Coder rows and the leak-free anchor/corpus counts (DBpedia 472/1881, Wikidata 1074/3869).
- Verify `razent/spbert-mlm-wso-base` resolves (alt: `razent/spbert-mlm-zero`).
