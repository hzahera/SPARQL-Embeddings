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

> ⚠ **Retrain both KGs after the `graph_builder.py` fixes.** The builder now
> encodes BIND constants, SELECT projections/aggregates, GROUP BY, LIMIT/OFFSET,
> MINUS/(NOT) EXISTS and SERVICE, and cleans property-path atoms so Wikidata
> label expansion reaches them. Existing checkpoints were trained on the old
> representation and cannot benefit. DBpedia graphs are unchanged by the fixes
> (every affected construct occurs in 0.0% of the DBpedia corpus), so a DBpedia
> retrain is a no-op re-run; Wikidata is the one that matters — exact
> graph-duplicates in its RQ1 corpus drop 24.9% → 6.9% and the largest collision
> class 216 → 18.

## Step 1b — RQ1 evaluation splits (already committed)

`datasets/{dbpedia,wikidata}-testsplit-RQ1.json` are the canonical RQ1 eval
splits: natural lexical distribution, invalid labels removed, and no
anchor–positive pair shared with the training groups. One identical protocol per
KG. See [datasets/RQ1_testsplits.README.md](datasets/RQ1_testsplits.README.md).

They are deterministic, so rebuilding reproduces them byte-for-byte (no oracle
needed):

```bash
python scripts/build_clean_rq1.py --kg dbpedia  --min-result-size 0 --out datasets/dbpedia-testsplit-RQ1.json
python scripts/build_clean_rq1.py --kg wikidata --min-result-size 0 --out datasets/wikidata-testsplit-RQ1.json
python scripts/check_leakage.py --kg wikidata --split datasets/wikidata-testsplit-RQ1.json   # expect CLEAN
```

## Step 2 — Fine-tune the E5 baseline

- `scripts/finetune_e5_baseline.py --kg {dbpedia,wikidata} --seed N` — scriptable,
  seed-parameterized form of the notebooks below; needed for the multi-seed
  significance test. Writes `checkpoints/e5_ft/<kg>/seed<N>/best_model`.
- `notebooks/finetuning/04_e5_contrastive_finetune_square.ipynb` (DBpedia → `checkpoints/e5_square_dbpedia/best_model`)
- `notebooks/finetuning/05_e5_contrastive_finetune_square_wikidata.ipynb` (Wikidata → `checkpoints/e5_square_wikidata/best_model`)

> **On early stopping.** Both trainers already select the **best-val-MRR
> checkpoint across epochs**, so an overfitted late-epoch model is never the one
> saved. `--patience` therefore only truncates the run to save compute — it can
> lose a late val improvement but cannot improve the selected checkpoint, and it
> is not a fix for overfitting (best-val selection already is one). Whatever you
> set, set the **same** `--patience` and `--min-delta` on both
> `train_sparqlce.py` and `finetune_e5_baseline.py`.
>
> The `graph_builder.py` fixes do **not** affect E5-FT (it reads raw query
> text), so existing E5-FT checkpoints stay valid. Retrain it for **seeds**, not
> for the fixes.

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
python scripts/build_hard_adversarial_rq1.py \
  --kg wikidata \
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

- `scripts/run_caching_experiment.py --kg {dbpedia,wikidata} --sparqlce <ckpt> --st ... --code-llm ... --e5-finetuned <ckpt> --min-answer-size 5`
  — the paper's current downstream-application experiment (semantic query
  caching). Headline setting: K=128, θ=0.8, `--min-answer-size 5` — see
  [results/caching/README.md](results/caching/README.md) for the full sweep
  and why 5 was chosen over 1/2/10.
- `notebooks/caching/caching_simulation_{dbpedia,wikidata}.ipynb` — earlier
  notebook form of the same experiment; the script above is the maintained
  path going forward.

## Step 6 — Significance testing (optional, for paper's significance tables)

One command does everything — trains 5 seeds of **both** SparqlCE and the
E5-Finetuned baseline, exports per-anchor scores for each, and runs the paired
test:

The eval splits are committed, so this is the whole procedure:

```bash
./scripts/run_seed_sweep.sh wikidata 5 datasets/wikidata-testsplit-RQ1.json 3
./scripts/run_seed_sweep.sh dbpedia  5 datasets/dbpedia-testsplit-RQ1.json  3
```

Notes:

- **The committed `{kg}-testsplit-RQ1.json` splits are the canonical ones.**
  The splits they replaced shared 84% of their
  anchor–positive pairs with the training groups (group-level split, but query
  text recurs across groups). That inflated SparqlCE's Wikidata lead from +0.06
  to +0.29 MRR@10 — see `datasets/RQ1_clean.README.md` → Leakage and
  `scripts/check_leakage.py`. The sweep now refuses a leaky split unless
  `ALLOW_LEAKY_SPLIT=1`.
- **Pull on the GPU box before running.** The sweep verifies from
  `run_metadata.json` that E5-FT trained at the batch-scaled LR
  (`1e-5 × batch/32`) and aborts if not: a pre-217fb8b checkout silently trains
  the baseline at 1e-5 on batch 128, costing it ~0.11 MRR@10.
- **Text baselines get label expansion + the E5 `"query: "` prefix** at eval
  (`eval_rq1_export.py --expand-labels/--text-prefix`, both `auto`). Without the
  first, a Wikidata text baseline ranks opaque `P31`/`Q5` ids while SparqlCE's
  graph builder resolves them internally; without the second, E5 is evaluated
  without the prefix it was trained with. `compute_significance.py` warns when
  the runs it aggregates predate this.
- **Two p-values are reported.** The anchor-level Wilcoxon treats the training
  run as fixed and cannot see cross-seed variance (SparqlCE ranged 0.571–0.881
  MRR@10 across seeds while its anchor-level p read 1e-32). The seed-level
  paired t-test is the one that reflects retraining instability, along with the
  distribution-free "worst SparqlCE seed vs best baseline seed" check.

- **Both sides get the same seeds.** `compute_significance.py --baseline-glob`
  compares **seed-averaged per-anchor scores**, so neither method is represented
  by one arbitrary training run. The old `--baseline` (single-run) path still
  works for back-compat but is weaker — prefer `--baseline-glob`.
- **Every run must use the same `--split`.** The paired Wilcoxon aligns
  per-anchor scores positionally; `compute_significance.py` now hard-errors on a
  split or anchor-list mismatch instead of silently producing a meaningless
  p-value.
- E5-FT is a full fine-tune saved as a standalone HF directory, so
  `--model-type sentence-transformer` is the correct export path for it. The
  `peft-qwen3` path only ever applied to bare LoRA adapter dirs and is now unused.

## Known gaps to close after runs

- `SUPPLEMENTARY.md` tables + paper rows: still labeled Qwen3; regenerate/rename with the E5-ft and Qwen2.5-Coder rows and the leak-free anchor/corpus counts (DBpedia 472/1881, Wikidata 1074/3869).
- `SUPPLEMENTARY.md` §4 hyperparameter table still says the text backbone is CodeBERT; `sparqlce/config.py` uses `intfloat/e5-base-v2`. Reconcile before submission — it also changes whether "SparqlCE beats its own fine-tuned backbone" is the right phrasing for the E5-FT comparison.
- Verify `razent/spbert-mlm-wso-base` resolves (alt: `razent/spbert-mlm-zero`).
