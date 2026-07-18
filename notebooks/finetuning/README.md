# Fine-tuning: capacity-matched E5-base-v2 baseline

Fine-tunes `intfloat/e5-base-v2` (~110M params) on SQuARE with a **standard
contrastive recipe** — in-batch InfoNCE, i.e. MultipleNegativesRankingLoss
(full fine-tune, AdamW). This is deliberately a *method-agnostic* adaptation
of a **capacity-matched** encoder (E5-base-v2 is roughly the size of
SparqlCE's own CodeBERT text encoder, ~125M) on our benchmark: it represents
"what a practitioner would do off the shelf with a comparable-scale model,"
so the gap to SparqlCE credits our contribution (the calibrated graded
objective + graph encoder), not backbone scale or the training recipe.

**Why E5-base-v2 and not Qwen3-Embedding-0.6B?** An earlier version of this
baseline fine-tuned Qwen3-Embedding-0.6B. But Qwen3 is ~5x larger and far
more heavily pretrained than SparqlCE's CodeBERT backbone, so a fine-tuned
Qwen3 beating SparqlCE conflates *backbone scale* with *training design* —
it doesn't isolate our contribution. E5-base-v2 holds backbone scale roughly
constant, so any remaining gap is attributable to SparqlCE's graph encoder +
calibrated objective. Qwen3 is still reported as a **zero-shot** baseline
(via `evaluate_rq1.py`) to show that scale alone, without our method, does
not close the gap. See `BASELINE_REDESIGN.md` for the full rationale.

**SparqlCE-specific components are excluded from this baseline:**
- no Jaccard-calibration term,
- no graph encoder,
- no variable-renaming augmentation (`AUGMENT_RENAME_PROB = 0.0`).

**Matched to SparqlCE's own training treatment:**
- **full fine-tune** (all backbone params), exactly as SparqlCE fully
  fine-tunes its own CodeBERT — not LoRA,
- fixed `LR = 1e-5` (SparqlCE's own CodeBERT LR), fixed 20 epochs, no LR
  sweep, no early stopping,
- effective batch 32 and scale 20 (== SparqlCE's τ=0.05), matched.

Training runs on SQuARE — using the benchmark for baselines is correct; a
benchmark is meant to be evaluated by every method. Only the *method* (loss +
architecture) is kept out of the baseline.

- `04_e5_contrastive_finetune_square.ipynb` — DBpedia.
- `05_e5_contrastive_finetune_square_wikidata.ipynb` — Wikidata.

## After re-running
The reported fine-tuned-baseline rows change from `Qwen3-Emb (fine-tuned)` to
`E5-base-v2 (fine-tuned)`. Regenerate: RQ1 tables, RQ2 robustness
(`notebooks/evaluation-RQ2/{DBpedia,Wikidata}/04b_eval_e5_finetuned_RQ2_*.ipynb`),
caching/compaction, and the per-model ρ for Fig. 3. See
`BASELINE_REDESIGN.md` for the full credit map and to-do list.
