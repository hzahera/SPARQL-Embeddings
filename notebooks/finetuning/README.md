# Fine-tuning: Qwen3 baseline

Fine-tunes `Qwen/Qwen3-Embedding-0.6B` on SQuARE with a **standard
contrastive recipe** — in-batch InfoNCE, i.e. MultipleNegativesRankingLoss
(LoRA, AdamW). This is deliberately a *method-agnostic* adaptation of a
strong LLM embedder on our benchmark: it represents "what a practitioner
would do off the shelf," so the gap to SparqlCE credits our contribution
(the calibrated graded objective + graph encoder), not the training recipe.

**SparqlCE-specific components are excluded from this baseline:**
- no Jaccard-calibration term (`LAMBDA_REG = 0.0`),
- no graph encoder,
- no variable-renaming augmentation (`AUGMENT_RENAME_PROB = 0.0`).

Training still runs on SQuARE — using the benchmark for baselines is
correct; a benchmark is meant to be evaluated by every method. Only the
*method* (loss + architecture) is kept out of the baseline.

- `04_qwen3_contrastive_finetune_square.ipynb` — DBpedia.
- `05_qwen3_contrastive_finetune_square_wikidata.ipynb` — Wikidata.

## Reproducing a calibrated *control* (optional)
To get the earlier supervision+calibration-matched variant (e.g. for an
architecture ablation, not as the headline baseline), set `LAMBDA_REG =
0.3` in the config cell; the loss block re-enables the Jaccard-MSE term
automatically.

## After re-running
The reported `Qwen3-Emb (fine-tuned)` rows change (expect them to drop vs.
the old calibrated version — a stronger story for SparqlCE). Regenerate:
RQ1 tables, RQ2 robustness, caching, and the per-model ρ for Fig. 3.
See `src-code/BASELINE_REDESIGN.md` for the full credit map and to-do list.
