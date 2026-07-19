# SparqlCE — paper plan

## Positioning (one line)
Existing SPARQL/text encoders capture **lexical** similarity; they fail to place
**result-equivalent but surface-divergent** queries close together. We contribute
a **result-aware** contrastive encoder (SparqlCE) + the **SQuARE** benchmark, and
show the gap is real and that answer-set supervision closes it.

## Research questions
- **RQ1 — Do textual/code SPARQL encoders capture result-equivalence, or only lexical similarity?**
  (Show they collapse on answer-equivalent / lexically-divergent queries.)
- **RQ2 — Does result-aware contrastive learning embed result-equivalent queries closer, especially when they are lexically dissimilar?**
  (Main result: stratified retrieval; SparqlCE wins where lexical fails.)
- **RQ3 — Are result-aware embeddings robust to answer-preserving perturbations?**
  (Invariance to renaming, FILTER(true), reordering, etc.)
- **RQ4 (optional) — Do they help a downstream task (semantic caching)?**

## Datasets
- **SQuARE** (ours): LSQ-derived query logs, **DBpedia + Wikidata**. Label = answer-set Jaccard.
- **Splits:** fixed, **leak-free** — evaluate only on held-out `test_idx` (0 anchor overlap with train/val). This is non-negotiable; the old sets leaked 66–84%.
- **Positives:** answer-Jaccard > 0.8. **Negatives:** lexically-similar / answer-different (hard).
- **Reporting:** stratify every result by anchor–positive lexical similarity (TF-IDF cosine) bins
  `[0,0.3) [0.3,0.5) [0.5,0.8) [0.8,1.0)`. **No threshold cherry-picking** — the crossover across bins is the figure.

## Baselines
- **Lexical:** TF-IDF, BM25
- **Text encoders:** SBERT-MiniLM, SBERT-MPNet, E5-base-v2
- **Code/SPARQL encoders:** CodeBERT, GraphCodeBERT, SPBERT  *(prior-literature SPARQL/code)*
- **Code LLM:** Qwen2.5-Coder-0.5B (zero-shot, last-token)
- **Fair fine-tuned control (optional):** E5-base fine-tuned with generic InfoNCE (capacity-matched to CodeBERT) — the "is it the data or the method?" control.

## Metrics
- **Retrieval (RQ1/RQ2):** MRR, Hits@k, nDCG@k — **per lexical bin** + aggregate.
- **Robustness (RQ3):** cosine invariance + MRR drop under 7 answer-preserving perturbations.
- **Downstream (RQ4):** caching **utility = hit-rate × result-quality** (single reconciled number).
- **Significance:** paired one-sided Wilcoxon vs the strongest baseline, per bin.

## Approach — improvements (priority order)
1. **Fix training first (highest ROI).** Both KG checkpoints must be well-converged — Wikidata is undertrained (val 0.35 vs DBpedia 0.45). This alone likely lifts the whole curve and may extend the win past the hardest bin.
2. **Make the result-aware objective central.** Graded regression of `cos(a,p)` onto answer-set Jaccard (continuous target), not just a small auxiliary term. This *is* the differentiator — lean into it.
3. **Harder negatives.** Upweight/sample more lexically-similar / answer-different negatives so the model explicitly learns lexical ≠ semantic.
4. **Stronger backbone.** CodeBERT is dated and already beaten (zero-shot) by GraphCodeBERT in the hard bin — try GraphCodeBERT or a stronger code encoder as the text side.
5. **Curriculum.** Train easy→hard (high→low lexical positives) so result-equivalence is learned progressively.
6. **Ablations (required for the story).** `−graph`, `−calibration`, `−hard-negatives` — isolate that the *result-aware objective*, not just the architecture, drives the gain.

## Honest guardrails
- Claim = "best in the surface-divergent regime, where all baselines collapse," **not** "best retriever overall" (lexical wins the easy bins / aggregate).
- Report the full stratified table; never a single hand-picked threshold.
- Fix the checkpoint, not the test set, when numbers look weak.
- Lead with DBpedia (stronger checkpoint, cleaner crossover); report Wikidata honestly alongside.
