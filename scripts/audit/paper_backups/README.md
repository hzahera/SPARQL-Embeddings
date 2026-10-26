# SQuARE paper — Overleaf project

Upload the contents of the supplied ZIP as a new Overleaf project. Select
`main.tex` as the main document and **pdfLaTeX** as the compiler. The project
uses Springer LNCS and includes its class and bibliography style locally.
TikZ figures and plot data are included; no shell escape or external assets
are needed. The TeX Live packages used are standard Overleaf packages.

Local build:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

This is the revised working manuscript, not a claim that all experiments are
complete. Red author TODOs remain visible deliberately. Before submission,
finish the baseline/full-corpus/generalization experiments; verify answer
canonicalization and cross-split identifiers; recompute reuse accuracy and
uncertainty from unrounded replay counts; verify licenses and the anonymous
artifact link; check the ESWC 2027 page and appendix limits.

Accuracy in the current reuse table is an approximate ratio reconstructed
from rounded mean rates, not a new raw-log computation. No missing results or
confidence intervals were fabricated. Threshold selection now has one
consistent validation optimization rule, with test accuracy reported
separately. CRC is described as a heuristic without a formal replay bound.

Only live manuscript dependencies are included. Dated drafts and the research
code remain outside this upload package. Build files and the compiled PDF
are excluded from the source ZIP. The final PDF is delivered separately.

The confirmed training objective uses anchor–positive–negative triplets and
multiple-negatives ranking: each anchor compares its positive against all B
positives and all B explicit negatives in the batch. Equation 2 matches this
B-by-2B objective; no Jaccard regression term is included. Cross-triplet
negatives are not claimed to be execution-verified for every anchor.

Reviewer-driven revision checklist (red TODOs are located beside the relevant
method or claim in main.tex):

1. Run lexical, algebra and query-form baselines on identical candidates.
2. Complete Tables/compact/fullcollection.tex and promote full held-out
   retrieval to the main result; keep sampled retrieval as a diagnostic.
3. Publish split audits and component/template statistics.
4. Audit actual training batches for duplicates and false negatives; run a
   masking ablation only if warranted, reporting it separately.
5. Compute paired component-bootstrap confidence intervals.
6. Finalize canonicalization, truncation policy, snapshots and training details.
7. Complete matched syntax supervision and the beyond-syntax diagnostic before
   claiming a causal label advantage or demonstrated learning beyond syntax.

Caching remains a fixed-threshold limitation study. A positive application
claim additionally needs validation-selected thresholds, accuracy/coverage
comparisons and incremental gains beyond exact caching on repeated workloads.
No new experiments were run in this manuscript revision; all new result cells
remain red placeholders. The original title, overview figure and confirmed
anchor-positive-negative objective are retained.
