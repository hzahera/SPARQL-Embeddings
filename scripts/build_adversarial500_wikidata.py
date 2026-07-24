#!/usr/bin/env python3
"""Build the purely-adversarial Wikidata RQ1 set (cos<0.3, >=3 positives/anchor).

The paper's core claim isolated: every anchor is paired ONLY with result-equivalent
positives whose lexical overlap with the anchor is negligible, so lexical / text
methods have no signal and only result-aware retrieval can succeed. Unlike the
stratified split (one positive/anchor), each anchor here carries >=3 such positives.

Spec
----
* Source: held-out TEST + VALID splits of wikidata-triplets-dataset.json (train
  excluded -> no training leakage). Valid is used only for model selection in the
  pipeline, so it is safe as evaluation data here.
* Positive: answer_jaccard > 0.8 (result-equivalent) AND anchor-positive TF-IDF
  cosine < 0.30 (lexically divergent). Every qualifying positive is kept.
* Anchor kept only if it has >= MIN_POS such positives.
* 500 anchors sampled deterministically (seed 42) from the eligible pool.
* NEG_PER hard negatives/anchor: lexically nearest queries (mined from the full
  triplets corpus) that are NOT result-equivalent positives of the anchor.

Lexical similarity is TF-IDF over raw SPARQL text (word 1-2 grams), fit on the
source anchors+positives -- identical basis to evaluate_rq1._lexical_cos_fn and
the other RQ1 splits.

Output: datasets/wikidata_testset_RQ1_adv500.json

    python scripts/build_adversarial500_wikidata.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

ROOT = Path(__file__).resolve().parents[1]
TRIP = ROOT / "datasets" / "wikidata-triplets-dataset.json"
SPLITS = ROOT / "datasets" / "splits_wikidata.json"
OUT = ROOT / "datasets" / "wikidata_testset_RQ1_adv500.json"

COS_MAX = 0.30
JAC_MIN = 0.80
MIN_POS = 3
N_ANCHORS = 500
NEG_PER = 10
SEED = 42

_norm = lambda s: " ".join(s.split())


def main():
    rng = np.random.default_rng(SEED)
    trip = json.load(open(TRIP))
    spl = json.load(open(SPLITS))

    # ---- source = held-out test + valid (train excluded) ----
    src_idx = list(spl["test_idx"]) + list(spl["valid_idx"])
    src = [trip[i] for i in src_idx]

    # ---- jaccard LUT + full-corpus mining pool (all splits, real queries) ----
    jac, pool = {}, set()
    for g in trip:
        a = _norm(g["anchor_query"])
        pool.add(a)
        for key in ("positive_queries", "negative_queries"):
            for p in g.get(key, []):
                q = p.get("query")
                if q is None:
                    continue
                qn = _norm(q)
                pool.add(qn)
                if p.get("answer_jaccard") is not None:
                    jac[(a, qn)] = float(p["answer_jaccard"])
    corpus_pool = sorted(pool)

    # ---- binning TF-IDF: fit on SOURCE anchors + positives ----
    btexts = set()
    for e in src:
        btexts.add(_norm(e["anchor_query"]))
        for p in e["positive_queries"]:
            btexts.add(_norm(p["query"]))
    btexts = list(btexts)
    bidx = {q: i for i, q in enumerate(btexts)}
    Xb = sk_normalize(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2)).fit_transform(btexts),
        norm="l2", axis=1,
    )
    cos = lambda a, q: float(Xb[bidx[a]].multiply(Xb[bidx[q]]).sum())

    # ---- eligible anchors: >= MIN_POS result-equivalent adversarial positives ----
    eligible = {}   # anchor_raw -> list of (pos_raw, cos, jac)
    for e in src:
        a_raw = e["anchor_query"]
        a = _norm(a_raw)
        advs = []
        for p in e["positive_queries"]:
            j = p.get("answer_jaccard")
            if j is None or j <= JAC_MIN:
                continue
            c = cos(a, _norm(p["query"]))
            if c < COS_MAX:
                advs.append((p["query"], c, float(j)))
        if len(advs) >= MIN_POS:
            eligible[a_raw] = advs
    print(f"eligible anchors (>= {MIN_POS} adversarial positives): {len(eligible)}")

    # ---- deterministic sample of N_ANCHORS ----
    keys = sorted(eligible)
    take = min(N_ANCHORS, len(keys))
    chosen = list(rng.choice(keys, size=take, replace=False))
    print(f"sampled {len(chosen)} anchors (target {N_ANCHORS})")

    # ---- mining TF-IDF over the full corpus pool (for hard negatives) ----
    ridx = {q: i for i, q in enumerate(corpus_pool)}
    X = sk_normalize(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2)).fit_transform(corpus_pool),
        norm="l2", axis=1,
    )

    out = []
    npos_list, negcos_list = [], []
    for a_raw in chosen:
        a = _norm(a_raw)
        advs = sorted(eligible[a_raw], key=lambda t: t[1])   # hardest first
        pos_texts = {_norm(q) for (q, _, _) in advs}
        pos_texts.add(a)

        # hard negatives: lexical nearest in the pool that are not positives
        sims = (X @ X[ridx[a]].T).toarray().ravel()
        hard = []
        for i in np.argsort(-sims):
            q = corpus_pool[i]
            if q in pos_texts:
                continue
            if jac.get((a, q), 0.0) > JAC_MIN:      # never admit a result-equivalent query
                continue
            hard.append({"query": q, "answer_jaccard": jac.get((a, q)),
                         "neg_type": "hard", "lexical_cos": round(float(sims[i]), 4)})
            if len(hard) >= NEG_PER:
                break

        out.append({
            "anchor_query": a_raw,
            "positive_queries": [
                {"query": q, "answer_jaccard": round(j, 4), "lexical_cos": round(c, 4)}
                for (q, c, j) in advs
            ],
            "negative_queries": hard,
            "lexical_bin": "low",                    # all cos < 0.3
            "n_positive": len(advs),
            "pos_cos_mean": round(float(np.mean([c for _, c, _ in advs])), 4),
            "pos_jaccard_min": round(float(min(j for _, _, j in advs)), 4),
        })
        npos_list.append(len(advs))
        negcos_list += [n["lexical_cos"] for n in hard]

    out.sort(key=lambda r: r["pos_cos_mean"])
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)

    allcos = [p["lexical_cos"] for r in out for p in r["positive_queries"]]
    print(f"\nwrote {len(out)} anchors to {OUT}")
    print(f"  positives/anchor: min={min(npos_list)} mean={np.mean(npos_list):.2f} "
          f"max={max(npos_list)}  total={sum(npos_list)}")
    print(f"  positive lexical cos: min={min(allcos):.3f} mean={np.mean(allcos):.3f} "
          f"max={max(allcos):.3f}  (all < {COS_MAX})")
    print(f"  positive jaccard min: {min(p['answer_jaccard'] for r in out for p in r['positive_queries']):.3f}")
    print(f"  hard negatives/anchor: {NEG_PER}  (lexical cos mean={np.mean(negcos_list):.3f})")
    corpus = set()
    for r in out:
        corpus.add(_norm(r["anchor_query"]))
        corpus.update(_norm(p["query"]) for p in r["positive_queries"])
        corpus.update(_norm(n["query"]) for n in r["negative_queries"])
    print(f"  unique corpus queries: {len(corpus)}")


if __name__ == "__main__":
    main()
