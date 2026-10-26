# Data audits (ESWC 2027, week 1)

Run from this folder with `PYTHONHASHSEED=0` (rdflib builds some lists from sets).

| Script | Needs | Gives |
|---|---|---|
| audit_splits.py <splits_dir> <graph> audit_<graph>.json | rdflib, networkx | cross-split leakage, components, templates, J=1 share, arity, query features |
| template_groups.py <splits_dir> <graph> | audit output | feasibility of a template-disjoint split |
| batch_audit.py <splits_dir> <graph> | split files | share of in-batch candidates that could be hidden positives |
| find_dups.py <splits_dir> <graph> | audit output | algebra-level matches across splits |
| table1_from_cache.py <cache.jsonl> | execution cache (server) | execution/labeling time, truncation, blank nodes |
| token_lengths.py (from repo root) | HF tokenizers (server) | queries over 512 tokens |

`main.before-w1b.tex` is a backup of the paper before the second round of W1 edits.
