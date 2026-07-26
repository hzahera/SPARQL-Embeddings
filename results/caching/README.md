# Semantic query-cache simulation — results

Produced by [`scripts/run_caching_experiment.py`](../../scripts/run_caching_experiment.py)
(protocol details: [`notebooks/caching/README.md`](../../notebooks/caching/README.md)).
FIFO cache over the K most-recent *executed* queries; incoming query → max
cosine similarity to cache; ≥ θ ⇒ hit (return cached answer set, do not
insert); < θ ⇒ miss (execute via the oracle, insert). Mean ± std over 5 seeds
(stream shuffle order only — embeddings are computed once per model).

## Headline setting: `--min-answer-size 5`

The held-out test streams are answer-set-size skewed: **49% (Wikidata) / 46%
(DBpedia)** of oracle-covered queries have exactly **one** true answer. For a
singleton answer set, Jaccard quality is binary (1 if the cache returns the
exact same entity, 0 otherwise) — there's no partial credit, so pooling these
queries with genuinely multi-answer queries into one mean-quality number
conflates two different regimes.

`--min-answer-size N` drops stream queries whose true answer set has fewer
than `N` rows before the simulation runs. We swept `N ∈ {1 (off), 2, 5, 10}`
on both KGs to pick a defensible headline setting rather than the
best-looking one — see the full sweep below. **N=5** is reported in the paper
because:

- It's the highest threshold that stays **uncontaminated on both KGs**. The
  three collapsed zero-shot encoders (E5-base-v2, GraphCodeBERT,
  Qwen2.5-Coder — see below) have no real signal and should score ~0 quality;
  on DBpedia their score drifts upward with the threshold (0.005 → 0.033 →
  0.067 → **0.174** at N=1/2/5/10) because restricting to larger answer sets
  raises the *chance* overlap between unrelated queries in DBpedia's smaller,
  more clustered answer universe. At N=5 that floor (0.067) is still ~3×
  below SparqlCE (0.201); at N=10 it isn't (0.174 vs 0.262) — the margin
  there is a floor artifact, not a finding. Wikidata's floor never drifts
  (stays ≈0.001–0.002 at every N), so N=5 is conservative there, not required.
- One threshold applied uniformly to both KGs avoids picking a different
  "best" cutoff per KG, which would be indistinguishable from cherry-picking.
- It still removes the singleton binary-Jaccard noise the filter exists for,
  without needing to justify an unusually high cutoff.

`quality` below is mean answer-set Jaccard over hits only; `hit×quality` is
the expected useful-hit value per incoming query (hit rate × quality),
included here because raw hit rate alone is misleading once a model has
collapsed (see next section) — bold = best per column, per KG.

### Wikidata (K=128, θ=0.8, min-answer-size=5, n=279)

| model | hit rate | quality | hit×quality |
|---|---|---|---|
| E5-base-v2 † | 0.996 ± 0.000 | 0.001 ± 0.002 | 0.001 |
| GraphCodeBERT † | 0.996 ± 0.000 | 0.001 ± 0.002 | 0.001 |
| Qwen2.5-Coder-0.5B † | 0.984 ± 0.000 | 0.001 ± 0.002 | 0.001 |
| SBERT/MPNet † | 0.946 ± 0.005 | 0.029 ± 0.015 | 0.027 |
| **SparqlCE (ours)** | 0.500 ± 0.002 | 0.124 ± 0.019 | **0.062** |
| E5-Finetuned | 0.449 ± 0.007 | 0.059 ± 0.017 | 0.026 |
| TF-IDF | 0.079 ± 0.007 | 0.275 ± 0.027 | 0.022 |
| BM25 | 0.073 ± 0.008 | 0.230 ± 0.047 | 0.017 |

### DBpedia (K=128, θ=0.8, min-answer-size=5, n=115)

| model | hit rate | quality | hit×quality |
|---|---|---|---|
| E5-base-v2 † | 0.991 ± 0.000 | 0.068 ± 0.051 | 0.067 |
| GraphCodeBERT † | 0.991 ± 0.000 | 0.068 ± 0.051 | 0.067 |
| Qwen2.5-Coder-0.5B † | 0.991 ± 0.000 | 0.068 ± 0.051 | 0.067 |
| SBERT/MPNet † | 0.909 ± 0.007 | 0.139 ± 0.067 | 0.126 |
| E5-Finetuned | 0.284 ± 0.004 | 0.639 ± 0.022 | 0.181 |
| **SparqlCE (ours)** | 0.307 ± 0.010 | 0.654 ± 0.013 | **0.201** |
| TF-IDF | 0.105 ± 0.000 | 0.427 ± 0.042 | 0.045 |
| BM25 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 |

† zero-shot, off-the-shelf. LaTeX: [`tab-caching-headline.tex`](../../tab-caching-headline.tex).

---

## Diagnosis: three zero-shot encoders are collapsed, not competitive

E5-base-v2, GraphCodeBERT, and Qwen2.5-Coder-0.5B report **identical**
hit rate and quality to 3 decimal places at every threshold tested. That's
not a coincidence: at ~99% hit rate they are effectively caching "everything,
always," so their measured quality is just the corpus's background
answer-overlap rate, independent of which embedding produced it — i.e. they
carry no usable signal for this task. Their apparent "advantage" on raw hit
rate is the opposite of useful for a cache: it means they can't discriminate
result-equivalent queries from unrelated ones at θ=0.8. Read their `hit×quality`
column, not their hit rate, when comparing to SparqlCE/E5-Finetuned/lexical.

## Full sweep (robustness check, not the headline)

`hit×quality`, K=128, θ=0.8, across every `--min-answer-size` tested:

**Wikidata**

| threshold | SparqlCE | E5-Finetuned | floor (collapsed encoders) |
|---|---|---|---|
| off (n=617) | 0.052 | 0.037 | ~0.001 |
| ≥2 (n=313) | 0.048 | 0.021 | ~0.000 |
| ≥5 (n=279) | **0.062** | 0.027 | ~0.001 |
| ≥10 (n=246) | 0.070 | 0.027 | ~0.002 |

**DBpedia**

| threshold | SparqlCE | E5-Finetuned | floor (collapsed encoders) |
|---|---|---|---|
| off (n=396) | 0.165 | 0.124 | 0.005 |
| ≥2 (n=212) | 0.189 | 0.157 | 0.033 |
| ≥5 (n=115) | **0.201** | 0.181 | 0.067 |
| ≥10 (n=70) | 0.262 | 0.250 | 0.174 ⚠ contaminated, see above |

SparqlCE leads at every threshold on both KGs; the margin over E5-Finetuned
narrows as the threshold rises (DBpedia: +33% → +20% → +10% → +5%) in step
with the rising floor — consistent with the floor-inflation mechanism above,
not a sign the models are converging in ability.

`BM25` at low hit rates (DBpedia ≥2: 0.008, ≥10: 0.000) is a handful of hits
or none at all — its quality/std numbers at those settings are noise, not a
measurement, and shouldn't be read as a result.

## Reproduce

```bash
python scripts/run_caching_experiment.py --kg wikidata \
  --sparqlce checkpoints_b128/wikidata/seed42_full/best_model.pt \
  --st intfloat/e5-base-v2 --st sentence-transformers/all-mpnet-base-v2 \
  --st microsoft/graphcodebert-base --code-llm Qwen/Qwen2.5-Coder-0.5B \
  --e5-finetuned checkpoints/e5_square_wikidata/best_model \
  --min-answer-size 5

python scripts/run_caching_experiment.py --kg dbpedia \
  --sparqlce checkpoints/sparqlce_dbpedia/best_model.pt \
  --st intfloat/e5-base-v2 --st sentence-transformers/all-mpnet-base-v2 \
  --st microsoft/graphcodebert-base --code-llm Qwen/Qwen2.5-Coder-0.5B \
  --e5-finetuned checkpoints/e5_square_dbpedia/best_model \
  --min-answer-size 5
```

Outputs (gitignored): `caching_<kg>_grid.csv` (full K×θ grid),
`caching_<kg>_headline.csv`/`.tex`, `caching_<kg>_pareto.csv`/`.png`. Each run
overwrites the previous — copy files aside (e.g. `_unfiltered`, `_min5`,
`_min10` suffixes) if you want more than one threshold's outputs preserved
simultaneously.
