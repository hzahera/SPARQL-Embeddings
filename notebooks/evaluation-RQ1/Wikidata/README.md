# RQ1 Evaluation (Wikidata)

- `06_eval_sparqlce_wikidata.ipynb` — retrieval metrics (MRR@{3,5,10,20},
  nDCG@{3,5,10,20}) for SparqlCE vs. baselines on
  `datasets/wikidata_testset_RQ1.json`.
- `07_visualize_alignment_wikidata.ipynb` — hexbin plot of embedding cosine
  similarity vs. answer-set Jaccard overlap, per model; saves
  `results/cosine_vs_jaccard_wikidata.pdf`.
