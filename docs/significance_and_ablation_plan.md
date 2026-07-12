# Work plan: statistical significance (Fatal #1) and architectural ablation (Fatal #2)

Reviewer risk being addressed: every headline number in Tables 2-6 is a
single-seed point estimate, and the core claim "the graph encoder
contributes" is currently unverified (Limitations admits the ablation
hasn't been run). Both are rejection-level at WSDM. This plan runs real
experiments to close both gaps -- it is not a reframing of the existing
text.

## What "done" looks like

- Every headline RQ1/RQ2 number for the full model is reported as
  mean +/- std over >=5 seeds, per KG.
- A paired significance test (Wilcoxon signed-rank over per-anchor scores)
  against the strongest baseline (fine-tuned Qwen3), with a p-value, per KG.
- An ablation table with >=3 seeds per variant showing MRR@10/nDCG@10 for:
  full model vs. no-GNN (bag of node embeddings) vs. no-decomposition
  (whole query as one text node) -- i.e. whether message passing over the
  graph helps, and whether decomposing into a graph at all helps.

## Fatal #1 -- Seed variance + significance

**Root cause:** `notebooks/training/{train_dbpedia,train_wikidata}.ipynb`
hardcode `SEED = 42` and were only ever run once per KG.

**What was built this session:**

- `scripts/train_sparqlce.py` -- CLI version of the two notebooks,
  parameterized by `--seed` and `--kg`, so it can be looped instead of
  hand-run. Same hyperparameters/recipe as the notebooks (InfoNCE +
  0.3*Jaccard-MSE, differential LRs, best-val-MRR checkpoint selection).
- `scripts/eval_rq1_export.py` -- like `evaluate_rq1.py` but exports
  **per-anchor** MRR@{3,5,10,20} and nDCG@{3,5,10,20}, not just the
  aggregate mean. Per-anchor scores are required for a paired test; the
  aggregate alone can't support one.
- `scripts/compute_significance.py` -- given N per-seed exports, computes
  cross-seed mean +/- std, picks a "representative" seed (closest to the
  cross-seed mean -- not the best seed, to avoid cherry-picking), and runs
  a one-sided Wilcoxon signed-rank test against a baseline's per-anchor
  scores on the same anchors.
- `scripts/run_seed_sweep.sh` -- orchestrates the above across N seeds x
  2 KGs.

**What you need to run (not run by me -- no GPU/torch in this environment):**

1. Export per-anchor scores for the strongest baseline (fine-tuned Qwen3)
   once per KG, so `compute_significance.py` has something to compare
   against:

   ```bash
   python scripts/eval_rq1_export.py --kg dbpedia \
       --model-type sentence-transformer --model-name <qwen3-finetuned path/id> \
       --ablation baseline-qwen3ft \
       --out results/significance/dbpedia_baseline_qwen3ft.json
   # repeat for wikidata
   ```

   If the fine-tuned Qwen3 isn't served through `sentence-transformers`
   (e.g. it's a LoRA-adapted checkpoint loaded some other way in the
   existing eval notebooks), swap in whatever loader those notebooks use
   -- the important part is producing the same
   `{anchors, aggregate, per_anchor}` JSON shape `eval_rq1_export.py`
   produces, since `compute_significance.py` only depends on that shape.

2. Run the sweep:

   ```bash
   ./scripts/run_seed_sweep.sh 5
   ```

   5 seeds x 2 KGs x 20 epochs. Cost: same as one full training run today,
   times 10. If today's DBpedia run took ~X hours, budget ~5X for DBpedia
   alone. Reduce to 3 seeds first if compute is tight -- 3 is a defensible
   minimum for reporting std, 5 is safer against reviewer pushback.

3. Read `results/significance/{dbpedia,wikidata}_full_summary.json` and
   send me the `cross_seed_stats` and `significance_vs_baseline` blocks --
   I'll turn them into the mean+-std table cells and a significance
   footnote/sentence for Tables 2-6 and the RQ1/RQ2 prose.

**Estimated new compute:** 10 additional full training runs (5 seeds x 2
KGs) versus the 2 you already ran. No new GPU-hours per run beyond what
today's DBpedia/Wikidata runs already cost.

## Fatal #2 -- Architectural ablation

**Root cause:** the paper's structural claim (graph topology + message
passing beats bag-of-text) was never isolated from the encoder's other
components; Limitations currently just admits this.

**What was built this session:**

- `sparqlce/encoder.py` -- `HybridSPARQLEncoder(ablation=...)` now supports:
  - `"full"` (default, unchanged): CodeBERT node/edge embeddings -> GATv2
    message passing -> mean pool -> project. This *is* today's model --
    no retraining needed for this row, reuse the seed-sweep checkpoints.
  - `"no_gnn"`: CodeBERT node embeddings, mean-pooled directly, GATv2
    skipped entirely. Isolates whether message passing over the graph
    topology adds anything beyond an unordered bag of the same node/edge
    text embeddings.
- `sparqlce/graph_builder.py` -- new `WholeQueryGraphBuilder`: represents
  each query as a single node holding the full canonicalized query
  string, no triple/FILTER/OPTIONAL decomposition. Paired with the `full`
  encoder mode (`--ablation no_decomposition` in the training script),
  this isolates whether decomposing into a graph at all helps, using the
  *same* backbone and training recipe as the full model -- a fairer
  comparison than the external text baselines (MiniLM/E5/Qwen3), which
  differ in both encoder and training procedure.
- `scripts/train_sparqlce.py --ablation {full,no_gnn,no_decomposition}` --
  same script as Fatal #1, the ablation is just another CLI flag.
- `scripts/run_ablation_sweep.sh` -- trains `no_gnn` and `no_decomposition`
  for 3 seeds each x 2 KGs (reuses the `full` seed-sweep checkpoints for
  the first row, so it doesn't retrain what you already have).
- `scripts/build_ablation_table.py` -- aggregates all three variants'
  per-seed exports into a LaTeX table (`tables/tab-ablation.tex`),
  mean +/- std, ready to `\input`.

**Known caveat to decide on, not silently defaulted:** `WholeQueryGraphBuilder`
reuses the existing node tokenizer, which truncates node labels at 32
tokens during training (`sparqlce/dataset.py:_to_data`, `max_length=32`)
and at only **16** tokens during inference/eval
(`sparqlce/embedder.py:_graph_to_data`, `max_length=16` -- this train/eval
mismatch already existed for the full model too, I didn't introduce it,
but it bites much harder here). For short queries this is fine; for
longer queries the no-decomposition baseline's single node may be cut off
well before the full query is seen, which would understate that baseline
and *overstate* the graph's advantage -- the ablation would then be
showing "decomposition beats truncation," not "decomposition beats
whole-query encoding." If your queries are often >16-32 tokens, raise
both caps for this ablation variant (and ideally reconcile the 16 vs. 32
mismatch for the full model too, as a separate fix) before trusting a
"no_decomposition loses badly" result. Happy to wire in a per-builder
token-length override if you want that instead of the shared default.

**What you need to run:**

```bash
./scripts/run_ablation_sweep.sh 3
```

Then send me `tables/tab-ablation.tex` (or let me read it directly) and
I'll write the paragraph that walks through it and update the
Limitations section to report a verified result instead of an admitted gap.

**Estimated new compute:** 2 ablation variants x 3 seeds x 2 KGs = 12
additional training runs, each roughly as expensive as one `full` run
(the no_gnn variant is actually cheaper -- no GATv2 forward/backward --
and no_decomposition trains on much shorter sequences per graph, so both
should undershoot the `full` run's wall-clock time, not match it).

## Sequencing

Run Fatal #1's seed sweep first -- it's the bigger rejection risk and the
`full`-ablation seeds it produces are reused (not wasted) by Fatal #2's
ablation table. Fatal #2 can start as soon as 3 of the 5 `full` seeds
are done; it doesn't need to wait for all 5.
