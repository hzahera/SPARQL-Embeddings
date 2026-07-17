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
2. Qwen3 + standard FT on SQuARE (MNR, no calibration) -> conventional
   adaptation doesn't match us (credits our METHOD).
3. SparqlCE -graph (same loss, CodeBERT flat) -> isolates the GRAPH.
4. SparqlCE (full).

## Consequences / to-do
- Re-run needed: current fine-tuned numbers change (likely drop -> stronger
  story). Regenerate RQ1/RQ2/caching rows + Fig. 3 rho for this baseline.
- Fig. 3: standard-loss baseline aligns geometry less, so its panel
  separates from SparqlCE — resolves the "look-alike" concern honestly.
- Wording: distinguish "answer-set supervision provided by SQuARE"
  (shared benchmark) from "our calibrated objective + graph encoder"
  (method) in Baselines (experiments-v2.tex) and Introduction.
