# Interactive demos

**Live:** [https://hzahera.github.io/SPARQL-Embeddings/](https://hzahera.github.io/SPARQL-Embeddings/) (GitHub shows `.html` files as code, so the pages are served with GitHub Pages from [`docs/`](../docs/)).

| Page | What it shows | Data |
|---|---|---|
| [`index.html`](https://hzahera.github.io/SPARQL-Embeddings/) | Overview and key numbers | paper tables |
| [`embeddings.html`](https://hzahera.github.io/SPARQL-Embeddings/embeddings.html) | Demo 1: the Figure 1 example, and linked 2-D maps of the real test-query embeddings | SQuARE test split |
| [`tasks.html`](https://hzahera.github.io/SPARQL-Embeddings/tasks.html) | Demo 2: retrieval explorer of real queries, Hits@1, cache hit accuracy and the threshold trade-off | SQuARE test split, paper tables |

## Embedding maps (Demo 1)

Each map places every test query (473 DBpedia, 968 Wikidata) with t-SNE. Click a query to see its nearest neighbour in each space and whether that neighbour returns the same answers.

```bash
# text-based (TF-IDF) and answer-overlap target spaces; CPU, ~30 s
python demo/scripts/build_embeddings.py

# add the real encoders: zero-shot vs. results-aware fine-tuned (needs torch + the trained checkpoint
# results/raw/<graph>/<model>/tuned/seed_<seed>/best_model); then the maps default to these two
python demo/scripts/build_embeddings.py --model e5-base-v2 --seed 42
```

## Rebuild the pages

```bash
python demo/scripts/build_demo_data.py   # retrieval examples (data/demo_data.js)
python demo/scripts/bundle_demo.py       # writes docs/index.html, docs/embeddings.html, docs/tasks.html
```

## Files

```text
demo/
├── index.html, embeddings.html, tasks.html   # page sources
├── assets/demo.css, demo.js, embmap.js       # styles and interactions (plain JavaScript, no libraries)
├── data/paper_results.js                     # numbers from the paper tables
├── data/demo_data.js                         # retrieval examples from the released splits
├── data/embeddings.js                        # 2-D embedding maps
└── scripts/                                  # build_demo_data.py, build_embeddings.py, bundle_demo.py
```

## Measured vs. illustrative

- **Measured:** the embedding maps, the retrieval explorer (TF-IDF ranking, real *J*), and all charts (paper tables).
- **Illustrative:** the two-dot sketch next to the Figure 1 queries.
