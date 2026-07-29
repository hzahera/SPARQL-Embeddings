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

## Step 1c — DBpedia rebuild from scratch (run this before any DBpedia step)

The DBpedia splits were deleted and are rebuilt on a different protocol. Run the
four commands in order; each writes what the next one reads. Everything except
the oracle is deterministic and takes seconds.

```bash
# 1. Execution oracle over the WHOLE corpus (4,727 queries, ~30-60 min).
#    The shipped result_sets_dbpedia.json covers only 1,677 (35.6%), which is
#    not enough to judge a pool exhaustively or to verify a hard negative.
python scripts/build_result_sets.py --kg dbpedia \
  --split datasets/dbpedia-triplets-dataset.json \
  --out datasets/result_sets_dbpedia_v2.json --pause 0.1

# 2. Component partition of the result-equivalence graph (replaces splits_dbpedia.json).
python scripts/build_component_splits.py --kg dbpedia \
  --result-sets datasets/result_sets_dbpedia_v2.json \
  --out datasets/splits_dbpedia_components.json

# 3. Held-out RQ1 splits: pooled + one file per axis, exhaustively judged.
python scripts/build_rq1_twoaxis.py --kg dbpedia \
  --splits datasets/splits_dbpedia_components.json \
  --result-sets datasets/result_sets_dbpedia_v2.json \
  --out datasets/dbpedia-testsplit-RQ1.json

# 4. Training and validation files, same rules, WITH hard negatives.
for s in train valid; do
  python scripts/build_rq1_twoaxis.py --kg dbpedia --from-split $s --neg-max 0.2 \
    --splits datasets/splits_dbpedia_components.json \
    --result-sets datasets/result_sets_dbpedia_v2.json \
    --out datasets/dbpedia-$s-RQ1.json
done
```

Pass the **same `--neg-max` to all three** (see "Choosing --neg-max" below), or
train, valid and test stop meaning the same thing.

Then train both models against the per-split files — **not** `--kg dbpedia`
alone, which addresses the old group split:

```bash
python scripts/train_sparqlce.py --kg dbpedia --seed 42 \
  --train-file datasets/dbpedia-train-RQ1.json \
  --val-file   datasets/dbpedia-valid-RQ1.json \
  --test-file  datasets/dbpedia-testsplit-RQ1.json \
  --hard-negatives --max-neg 16

python scripts/finetune_e5_baseline.py --kg dbpedia --seed 42 \
  --train-file datasets/dbpedia-train-RQ1.json \
  --val-file   datasets/dbpedia-valid-RQ1.json
```

Both **must** get the same `--train-file`. Training SparqlCE on the component
split while the baseline keeps the old group split would compare two models over
different corpora, and the old split's training anchors sit inside the new test
components.

> **Negative parity — do not omit `--hard-negatives`.**
> `finetune_e5_baseline.py` *always* places one curated negative per anchor in
> its InfoNCE candidate set. `train_sparqlce.py` ignores `negative_queries`
> unless `--hard-negatives` is passed. With the old corpus this was harmless (24
> lexically-unrelated negatives shared across 4,713 groups); with the component
> files it is not, because they carry ~16 same-template, result-disjoint
> negatives per anchor. Omitting the flag trains the **baseline** against exactly
> the lexical-vs-result distinction SparqlCE is meant to capture, and SparqlCE
> without it. `run_seed_sweep.sh` now defaults it on in per-split mode
> (`HARD_NEGS=0` to disable deliberately).
>
> A first single-seed run made this mistake: E5-FT reached MRR@10 0.965 against
> SparqlCE's 0.889. Do not compare against those numbers.

Turning the flag on changes the memory profile, so the batch size has to change
with it:

> **Batch size with `--hard-negatives`.** The flag sends `BATCH x MAX_NEG`
> negative *graphs* through the encoder per step, on top of `BATCH` anchors and
> `BATCH` positives, each graph costing a BERT pass per node. At the sweep's
> default `128 x 16 = 2048` an 80GB A100 OOMs on step 1; in-batch mode encodes
> 256 graphs total. Their **product** is the budget. `run_seed_sweep.sh` refuses
> to start above `MAX_NEG_GRAPHS` (default 512).
>
> Use `BATCH=32`, `MAX_NEG=8`. Candidates per anchor then work out to 40 for
> SparqlCE (32 in-batch + 8 hard) against 64 for E5-FT (`2 x BATCH`, since
> `info_nce_loss` ranks against every positive *and* every negative in the
> batch). The baseline keeps slightly more, which is the safe direction: a
> SparqlCE win under it is conservative.
>
> ```bash
> MAX_NEG=8 ./scripts/run_seed_sweep.sh dbpedia 5 datasets/dbpedia-testsplit-RQ1.json 3 32 60
> ```

### Choosing `--neg-max`

`--neg-max` is where a candidate stops being "unjudged" and becomes a negative.
The default 0.0 (only fully disjoint queries are negatives) is far too strict on
DBpedia: answer sets are small and drawn from a shared vocabulary, so most pairs
share one row by coincidence, 94% of the test pool ends up unjudged, and each
anchor ranks ~15 candidates of which 12.6 are positives — a metric that cannot
separate two systems.

The measured overlap distribution is bimodal: **44% of test pairs at
`0.05 < J <= 0.1`** (two ~15-row answer sets sharing exactly one row —
coincidence, not partial relevance) and **43% at `0.5 < J <= 0.8`** (substantial
overlap, genuinely ambiguous). Between them is a valley, and on the **test**
split the `0.2-0.3` bin is empty, so any threshold in it yields a bit-identical
split. `--neg-max 0.2` is therefore not a tuned value on the split that gets
reported. The builder prints a sensitivity table making this checkable; note
that train and valid *do* have a little mass there (0.5% and 0.3%), so state the
insensitivity claim about the test split specifically.

The `0.5-0.8` band stays unjudged, which drops ~123 candidates per anchor from
each ranking. Disclose that: result-equivalence is not transitive, but connected
components treat it as if it were, so a component pool contains many
chain-related, partially-overlapping queries.

### What changed and why

- **`splits_dbpedia.json` is superseded.** 100% of the 39,520 positives are
  themselves anchors, so the corpus is one symmetric equivalence relation over
  4,713 queries, not 4,713 independent items. A group split therefore *cannot*
  be leak-free — that is the 84.2% pair leakage, not a bug in the old file.
  Partitioning connected components (1,118 of them, largest 161) makes a
  crossing pair impossible by definition, so `build_clean_rq1.py`'s R3 repair is
  no longer needed. The builder asserts 0 crossing pairs and refuses to write
  otherwise.
- **Judgments are exhaustive.** Every anchor is scored against every pool query
  with the oracle. Partial overlap (`0 < J <= 0.8`) becomes *unjudged*, not
  negative. Because the pool is drawn from the same split, a newly discovered
  positive cannot be a training pair — the old build had to throw away 547 such
  discoveries.
- **Negatives are hard.** The shipped triplets carry **24 distinct negatives for
  the entire corpus**, all lexically unrelated to their anchors, so nothing in
  the old benchmark penalised a purely lexical matcher. The new negatives share
  the anchor's structural template and differ only in constants: near-identical
  text, provably disjoint answers. 81.5% of same-template DBpedia pairs qualify.
  **This changes the training data**, so DBpedia checkpoints must be retrained,
  and it is the change most likely to move the headline number.
- **Two axes are reported separately.** `-axisS` (different template, same
  answers — a structural rewrite, a property of the query) and `-axisC` (same
  template, different constants with the same extension — a property of the
  graph, learnable only through knowledge about constants). In each file the
  other axis's positives are *excluded*, not scored as errors. Report both:
  pooling them hides which phenomenon a score came from, and axis C is where
  memorization would show up as skill.

Report the **pool size** next to any MRR: the builder prints it, along with how
many judgeable queries never reached the ranking.

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
