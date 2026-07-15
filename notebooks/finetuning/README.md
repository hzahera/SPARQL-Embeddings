# Fine-tuning: Qwen3 baseline

Fine-tunes `Qwen/Qwen3-Embedding-0.6B` on SQuARE (result-set-grounded SPARQL
query pairs with answer-set Jaccard supervision), reusing the SparqlCE
training recipe (InfoNCE in-batch contrastive loss + 0.3 × Jaccard-MSE
calibration term, AdamW). This produces the strongest baseline used in the
RQ1/RQ2 evaluations.

- `04_qwen3_contrastive_finetune_square.ipynb` — DBpedia.
- `05_qwen3_contrastive_finetune_square_wikidata.ipynb` — Wikidata.
