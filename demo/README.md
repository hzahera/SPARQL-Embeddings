# Interactive demo: Results-Aware SPARQL Embeddings

**Live page:** [https://hzahera.github.io/SPARQL-Embeddings/](https://hzahera.github.io/SPARQL-Embeddings/). GitHub shows `.html` files as code, so the page is also built as one self-contained file, [`docs/index.html`](../docs/index.html), which any static host can serve (Netlify, Anonymous GitHub "GitHub Pages", or GitHub Pages with folder `/docs`).

**After editing anything in `demo/`, rebuild the single file:**

```bash
python demo/scripts/bundle_demo.py      # writes docs/index.html
```

The page explains the paper in six short, interactive sections:

| Section | What you can do | Data |
|---|---|---|
| 1 · The problem | Switch between a text-based and a results-aware encoder on the three queries of Figure 1; see TF-IDF similarity vs. answer overlap for real test pairs | Figure 1, SQuARE test pairs |
| 2 · How it works | Step through the pipeline: compute *J* with a Venn diagram, compare a random split with our component split, and watch a toy contrastive training run live (InfoNCE + hard-negative mining) | toy examples |
| 3 · Query retrieval | Browse real test queries where text similarity ranks the wrong query first; compare zero-shot vs. fine-tuned Hits@1 for all ten encoders, the syntax-label control, and seen vs. unseen templates | SQuARE test split, paper tables |
| 4 · Result caching | Step through cache decisions; compare hit accuracy before and after fine-tuning; move the threshold θ along the measured risk-coverage curves | paper tables |
| 5 · What is new | Comparison with prior lines of work; SQuARE statistics | paper |
| 6 · Reproduce | Commands to rebuild the page data and the full benchmark | – |

## Files

```text
demo/
├── index.html                 # the page
├── assets/demo.css            # styles (light and dark mode)
├── assets/demo.js             # interactions and charts (plain JavaScript, no libraries)
├── data/demo_data.js          # real examples, generated from the released splits
├── data/paper_results.js      # numbers from the paper tables
├── scripts/build_demo_data.py # rebuilds data/demo_data.js
└── scripts/bundle_demo.py     # inlines everything into docs/index.html
```

## Rebuild the real-data parts

```bash
python demo/scripts/build_demo_data.py --repo . --out demo/data/demo_data.js
```

The script uses only the released files (`datasets/*-splits/*-train.json`, `*-test.json`, `*-test-rq3.json`). It ranks every test query against all other test queries with TF-IDF (fitted on the training split, as in `scripts/baselines/nonneural.py`) and judges each ranking with the complete answer-overlap labels. As a check, it reproduces the paper's TF-IDF Hits@1 with complete labels (0.60 vs. 0.58 on DBpedia, 0.37 vs. 0.37 on Wikidata; small differences come from random tie-breaking).

## What is measured and what is illustrative

- **Measured:** the scatter plot and the retrieval explorer (real test queries, real *J*, TF-IDF computed from the data), and every chart in sections 3 and 4 (numbers from the paper tables in `data/paper_results.js`).
- **Illustrative:** the embedding sketch in section 1, the toy training run in section 2 (12 free 2-D points, not a real encoder), and the cosine values in the cache walk-through. Each of these is labeled on the page.
