# RQ1 Evaluation (DBpedia)

- `06_eval_sparqlce_dbpedia.ipynb` — retrieval metrics (MRR@{3,5,10,20},
  nDCG@{3,5,10,20}) for SparqlCE vs. baselines on
  `datasets/dbpedia_testset_RQ1.json`.
- `07_visualize_alignment_dbpedia.ipynb` — hexbin plot of embedding cosine
  similarity vs. answer-set Jaccard overlap, per model; saves
  `results/cosine_vs_jaccard_dbpedia.pdf`.
