# Fine-tuned baseline redesign — crediting the contribution correctly

## The problem
The current `Qwen3-Emb (fine-tuned)` baseline is trained with **our** loss
(InfoNCE + Jaccard-calibration). That calibrated graded objective is part
of our contribution, so the baseline = "our method on their backbone."
This (a) leaks credit and (b) makes the baseline nearly match us on
geometry (the calibration term is what aligns cosine with Jaccard).

Note: training the baseline on **SQuARE is fine** — a benchmark is meant to
be used by all methods. The leak is the **loss**, not the data.

## The fix — fine-tune with a STANDARD recipe
Adapt Qwen3-Embedding with a conventional embedding-fine-tuning objective,
not ours:

- **MultipleNegativesRankingLoss (MNR)** in Sentence-Transformers, or plain
  InfoNCE, on the SQuARE anchor/positive pairs (in-batch + optional hard
  negatives). LoRA, same train split, comparable epochs/lr.
- **No Jaccard-calibration term** (ours). **No graph** (ours).
- Keep it strong — MNR is near-SOTA for this; do not handicap it.

Result: the baseline is "strong LLM embedder + standard fine-tuning on our
benchmark," and the remaining gap to SparqlCE credits our **calibrated
graded objective + graph encoder**.

## Keep the architecture isolation separate
The standard-recipe baseline no longer isolates the graph encoder. If we
want that (the ablation reviewers ask for), run it separately:
`SparqlCE -graph` = CodeBERT-flat (mean-pool) with our FULL loss, vs
CodeBERT+GAT. Same backbone, one variable.

## Clean credit map (target baseline/ablation set)
1. Qwen3 zero-shot            -> scale alone doesn't help.
2. E5-base-v2 + standard FT on SQuARE (generic InfoNCE, no calibration,
   full fine-tune) -> conventional adaptation of a *capacity-matched*
   encoder doesn't match us (credits our METHOD, controlling for scale).
3. SparqlCE -graph (same loss, CodeBERT flat) -> isolates the GRAPH.
4. SparqlCE (full).

## Revision: capacity-matched fine-tuned baseline (2026-07-18)
The fine-tuned baseline was **Qwen3-Embedding-0.6B** (generic InfoNCE, LoRA).
Problem: even after fixing data leakage and matching SparqlCE's own fixed
training schedule, fine-tuned Qwen3 *beats* SparqlCE on RQ1 (MRR ~0.75 vs
~0.53 on DBpedia). But Qwen3 is ~5x larger and far more heavily pretrained
than SparqlCE's CodeBERT backbone, so that result conflates **backbone
scale** with **training design** — it does not isolate our contribution and
invites the reviewer objection "you just used a bigger model."

**Fix:** replace the *fine-tuned* baseline with **E5-base-v2** (~110M,
comparable to CodeBERT's ~125M), full fine-tuned (matching SparqlCE's own
full-FT-of-CodeBERT treatment) with the same generic InfoNCE recipe. This
holds backbone scale roughly constant, so the SparqlCE-vs-baseline gap
credits the graph encoder + calibrated objective rather than model size.
Qwen3 remains as a **zero-shot** row (scale alone doesn't help).

- Fine-tuning notebooks: `notebooks/finetuning/04_e5_contrastive_finetune_square.ipynb`,
  `05_e5_contrastive_finetune_square_wikidata.ipynb` (Qwen3 finetuning
  notebooks removed).
- RQ2: `notebooks/evaluation-RQ2/{DBpedia,Wikidata}/04b_eval_e5_finetuned_RQ2_*.ipynb`
  (Qwen3-ft RQ2 notebooks removed).
- Removed stale Qwen3-ft artifacts: `results/significance/*_baseline_qwen3ft.json`,
  `results/robustness/robustness_qwen3_finetuned_*.json`.
- **Still to do (GPU re-runs):** train E5 on both KGs; regenerate RQ1/RQ2/
  caching/compaction rows and Fig. 3 ρ with the E5-ft row; update
  `scripts/eval_rq1_export.py` (the `peft-qwen3` significance path) for the
  E5 full-FT model; rename the fine-tuned row in `SUPPLEMENTARY.md` tables
  and the paper. Consider `SparqlCE -graph` (backbone held constant, graph
  ablated) as the complementary architecture isolation. See also the
  optional stronger control: swap SparqlCE's own CodeBERT encoder for
  Qwen3/E5 weights so both sides share a backbone.

## Consequences / to-do
- Re-run needed: current fine-tuned numbers change (likely drop -> stronger
  story). Regenerate RQ1/RQ2/caching rows + Fig. 3 rho for this baseline.
- Fig. 3: standard-loss baseline aligns geometry less, so its panel
  separates from SparqlCE — resolves the "look-alike" concern honestly.
- Wording: distinguish "answer-set supervision provided by SQuARE"
  (shared benchmark) from "our calibrated objective + graph encoder"
  (method) in Baselines (experiments-v2.tex) and Introduction.
