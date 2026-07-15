# Embedding space visualization

`01_visualization_dbpedia.ipynb` — for a sampled anchor query, projects the
SparqlCE and baseline (TF-IDF, MiniLM, MPNet, E5-base, Qwen3, fine-tuned
Qwen3) embedding spaces to 2D (TruncatedSVD/t-SNE) and plots candidates
colored by answer-set Jaccard overlap with the anchor, to visually contrast
how each embedding space reflects result-set similarity.

Requires a trained SparqlCE checkpoint under `sparqlce/models/` (not tracked
in git — see the main [README](../../README.md)).
