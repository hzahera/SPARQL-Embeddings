#!/usr/bin/env python3
"""Build a randomly-sampled Wikidata RQ1 set (representative, NOT adversarial).

Unlike adv500 (which selects anchors by cos<0.3), this samples anchors at random
from the held-out test split and keeps their natural result-equivalent positives
with NO lexical filter -- so the anchor-positive lexical similarity follows its
natural distribution (reported, not constrained). Representative difficulty,
higher/more believable absolute scores.

Spec
----
* Source: held-out TEST split only (no valid, no train -> no leakage).
* Positive: answer_jaccard > 0.8 (result-equivalent). All kept. No cos filter.
* Anchor kept only if it has >= MIN_POS such positives; 500 sampled (seed 42).
* NEG_PER hard negatives/anchor: lexically nearest queries (mined from the full
  triplets corpus) that are NOT result-equivalent positives of the anchor.

Output: datasets/wikidata_testset_RQ1_rand500.json

    python scripts/build_random500_wikidata.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

ROOT = Path(__file__).resolve().parents[1]
TRIP = ROOT / "datasets" / "wikidata-triplets-dataset.json"
SPLITS = ROOT / "datasets" / "splits_wikidata.json"
OUT = ROOT / "datasets" / "wikidata_testset_RQ1_rand500.json"

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
    src = [trip[i] for i in spl["test_idx"]]          # held-out test only

    # jaccard LUT + full-corpus mining pool
    jac, pool = {}, set()
    for g in trip:
        a = _norm(g["anchor_query"])
        pool.add(a)
        for key in ("positive_queries", "negative_queries"):
            for p in g.get(key, []):
                q = p.get("query")
                if q is None:
                    continue
                qn = _norm(q); pool.add(qn)
                if p.get("answer_jaccard") is not None:
                    jac[(a, qn)] = float(p["answer_jaccard"])
    corpus_pool = sorted(pool)

    # cos basis: fit on source anchors + positives (for reporting the natural cos)
    btexts = set()
    for e in src:
        btexts.add(_norm(e["anchor_query"]))
        for p in e["positive_queries"]:
            btexts.add(_norm(p["query"]))
    btexts = list(btexts); bidx = {q: i for i, q in enumerate(btexts)}
    Xb = sk_normalize(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2)).fit_transform(btexts),
        norm="l2", axis=1)
    cos = lambda a, q: float(Xb[bidx[a]].multiply(Xb[bidx[q]]).sum())

    # eligible: >= MIN_POS result-equivalent positives (NO cos filter)
    eligible = {}
    for e in src:
        a = _norm(e["anchor_query"])
        pos = [(p["query"], cos(a, _norm(p["query"])), float(p["answer_jaccard"]))
               for p in e["positive_queries"]
               if p.get("answer_jaccard") is not None and p["answer_jaccard"] > JAC_MIN]
        if len(pos) >= MIN_POS:
            eligible[e["anchor_query"]] = pos
    print(f"eligible anchors (>= {MIN_POS} result-equiv positives): {len(eligible)}")

    keys = sorted(eligible)
    chosen = list(rng.choice(keys, size=min(N_ANCHORS, len(keys)), replace=False))
    print(f"randomly sampled {len(chosen)} anchors (seed {SEED})")

    # mining TF-IDF over the full corpus pool
    ridx = {q: i for i, q in enumerate(corpus_pool)}
    X = sk_normalize(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2)).fit_transform(corpus_pool),
        norm="l2", axis=1)

    out, npos_list, negcos = [], [], []
    for a_raw in chosen:
        a = _norm(a_raw)
        pos = eligible[a_raw]
        pos_texts = {_norm(q) for (q, _, _) in pos}; pos_texts.add(a)
        sims = (X @ X[ridx[a]].T).toarray().ravel()
        hard = []
        for i in np.argsort(-sims):
            q = corpus_pool[i]
            if q in pos_texts or jac.get((a, q), 0.0) > JAC_MIN:
                continue
            hard.append({"query": q, "answer_jaccard": jac.get((a, q)),
                         "neg_type": "hard", "lexical_cos": round(float(sims[i]), 4)})
            if len(hard) >= NEG_PER:
                break
        out.append({
            "anchor_query": a_raw,
            "positive_queries": [
                {"query": q, "answer_jaccard": round(j, 4), "lexical_cos": round(c, 4)}
                for (q, c, j) in pos],
            "negative_queries": hard,
            "n_positive": len(pos),
            "pos_cos_mean": round(float(np.mean([c for _, c, _ in pos])), 4),
        })
        npos_list.append(len(pos)); negcos += [n["lexical_cos"] for n in hard]

    out.sort(key=lambda r: r["pos_cos_mean"])
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)

    allcos = [p["lexical_cos"] for r in out for p in r["positive_queries"]]
    print(f"\nwrote {len(out)} anchors to {OUT}")
    print(f"  positives/anchor: min={min(npos_list)} mean={np.mean(npos_list):.2f} "
          f"max={max(npos_list)} total={sum(npos_list)}")
    print(f"  positive lexical cos (NATURAL): min={min(allcos):.3f} "
          f"mean={np.mean(allcos):.3f} max={max(allcos):.3f}")
    for t in (0.3, 0.5, 0.8):
        print(f"    frac positives cos<{t}: {np.mean(np.array(allcos) < t):.1%}")
    print(f"  hard negatives/anchor: {NEG_PER} (lexical cos mean={np.mean(negcos):.3f})")


if __name__ == "__main__":
    main()
