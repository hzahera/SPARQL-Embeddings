# Training Scripts

- `generate_training_triplets.py`: generate training triplets from unique
  SPARQL queries using Jaccard similarity.

Actual model training is done via `scripts/train_sparqlce.py` at the repo
root (`--kg {dbpedia,wikidata} --seed N [--ablation ...]`), which replaces
the placeholder scripts that used to live in this directory.
