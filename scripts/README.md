# Scripts

Utility scripts for dataset generation, evaluation, training, and analysis.

## Dataset generation

- **`extract_unique_queries.py`** — extract unique SPARQL queries from a hard
  triplets JSON export (an intermediate dataset-construction artifact, not
  tracked in this repo).

  ```bash
  python scripts/extract_unique_queries.py --input hard_triplets_tfidf_dbpedia.json \
    --output unique_queries_dbpedia.json
  ```

- **`generate_hard_robustness_dataset.py`** — generate the DBpedia RQ2 perturbation dataset (`--input`/`--output`/`--seed`/`--types`).
- **`generate_robustness_wikidata_RQ2.py`** — generate the Wikidata RQ2 perturbation dataset.
- **`add_missing_perturbations.py`** — backfill missing perturbation types into an existing RQ2 dataset.
- **`training/generate_training_triplets.py`** — generate training triplets from unique queries via Jaccard similarity.

### RQ1 splits — the component protocol (DBpedia)

Run in order; see [Step 1c of RUN_GUIDE.md](../RUN_GUIDE.md) for the full runbook
and rationale.

- **`build_result_sets.py`** — the execution oracle. Pass a KG's
  `*-triplets-dataset.json` as `--split` to execute the **whole** corpus, which
  is what exhaustive judging and hard negatives need (the shipped DBpedia oracle
  covers 35.6%). Aborts early if the first `--abort-after` queries all fail,
  rather than spending hours to write an empty file.
- **`build_component_splits.py`** — partitions the *connected components* of the
  result-equivalence graph instead of anchor groups. Since 100% of DBpedia's
  positives are themselves anchors, a group split cannot avoid restating a
  training pair at eval time; a component split makes crossing pairs impossible,
  and the script asserts 0 before writing. Packs on query count **and** on
  axis-S/axis-C anchor counts, which are too lumpy for a one-pass greedy.
- **`build_rq1_twoaxis.py`** — the RQ1 split: exhaustive oracle judgments over
  the pool, same-template hard negatives, and per-axis files (`-axisS`,
  `-axisC`) alongside the pooled one. Also builds the **training** and
  validation files via `--from-split {train,valid}`, so all three splits get the
  same rules and the same hard negatives.

`build_clean_rq1.py` and `relabel_rq1_split.py` remain the path for a
**group-split** KG (currently Wikidata, whose oracle covers too little of its
corpus for the above). Do not mix protocols within one comparison.

## Training

- **`train_sparqlce.py`** — CLI training script (`--kg {dbpedia,wikidata} --seed N [--ablation {full,no_gnn,no_decomposition}]`),
  consolidating `notebooks/training/{train_dbpedia,train_wikidata}.ipynb` into something loopable for seed/ablation sweeps.
- **`run_seed_sweep.sh`** — trains N seeds per KG and aggregates significance stats; see `results/significance/README.md`.
- **`run_ablation_sweep.sh`** — trains the `no_gnn`/`no_decomposition` ablation variants and builds the ablation table.

## Evaluation

- **`verify_invariances.py`** — checks the graph builder's variable-renaming and `FILTER(true)` invariances.
- **`measure_fallback_fraction.py`** — measures how often the graph builder falls back to a flat token-chain representation.
- **`eval_rq1_export.py`** — RQ1 evaluation with per-anchor score export (needed for paired significance testing; see `notebooks/evaluation-RQ1/evaluate_rq1.py` for the aggregate-only version).
- **`compute_significance.py`** — cross-seed mean/std and a paired Wilcoxon signed-rank test against a baseline.
- **`build_ablation_table.py`** — aggregates per-seed ablation exports into a LaTeX table.
- **`run_caching_experiment.py`** — downstream semantic-caching simulation (`--kg {dbpedia,wikidata}`, model roster flags, `--min-answer-size`); see `results/caching/README.md`, the paper's current downstream-application experiment.

## Plotting

- **`plot_robustness_radar.py`** — per-operator robustness radar (paper appendix figure).
- **`plot_rank_stability.py`** — MRR-before/after-perturbation bar chart.
- **`plot_caching_pareto.py`** — utility-vs-theta figure from `run_caching_experiment.py`'s pareto CSVs (paper `fig:caching`).
