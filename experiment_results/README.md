# Experiment results

Result files behind every number in the paper: 10 encoders x 2 graphs, zero-shot and
five fine-tuned seeds (42-46), plus the syntax-supervised control (3 encoders x 3 seeds).
Model checkpoints are not included. The layout mirrors `results/`, the working folder
written by the scripts; to run the analysis scripts on these files, use
`ln -s experiment_results results` from the repository root.

| Paper | File(s) |
|---|---|
| Table 2 (stored labels), RQ1 per-encoder Spearman rho | `aggregated/summary_{dbpedia,wikidata}.json`; per run: `raw/<graph>/<model>/{zero_shot,tuned/seed_*}/evaluation_test.json` |
| Table 3 (complete labels), Sec. 4.3 paired CIs, seen/unseen templates | `analysis/paired_analysis_{dbpedia,wikidata}.json` (from `per_anchor/`) |
| Non-neural baselines (Tables 2, 3) | `analysis/baselines_{dbpedia,wikidata}.json`, `per_anchor/<graph>/baselines.json` (per anchor) |
| Table 4 (result caching), mean J of reused results | `raw/<graph>/<model>/*/cache.json`, `aggregated/summary_*.json` |
| Sec. 4.4 two-level bootstrap intervals | `analysis/cache_ci.json` |
| Threshold selection, Appendix B, Fig. 4 | `analysis/calibration_{dbpedia,wikidata}.json` |
| Checkpoint selection (nDCG@10 vs nDCG@5) | `analysis/checkpoint_selection.json` |
| Timing (Sec. 4.1) | `analysis/timing_labeling_{dbpedia,wikidata}.json`, `analysis/timing_inference_wikidata.json` (encoding and search, measured on Wikidata) |
| Syntax-supervised control (Table 3, Sec. 4.3) | `syntax_control/control_results.json`, `syntax_control/control_latex.txt`; per run: `syntax_control/raw/` |

Per run, `training_history.json` holds the validation scores of every epoch and
`best_checkpoint.json` the selected epoch.
