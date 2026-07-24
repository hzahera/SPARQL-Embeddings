#!/usr/bin/env python3
"""Build the stratified, hard-negative Wikidata RQ1 test split (Option B).

Motivation
----------
The standard RQ1 test set has three confounds that muddy the RQ1 claim
(result-aware retrieval beats lexical):
  1. positives/anchor ranges 1..20  -> P@k / R@k ceilings, macro-averages mix
     incompatible regimes;
  2. answer_jaccard is bimodal (singleton result sets) and ~45% of positives are
     lexical near-duplicates (cos >= 0.3) -> aggregate rewards lexical baselines;
  3. only 21 distinct negatives, reused across all anchors -> weak distractors.

This split fixes all three:
  * exactly ONE result-equivalent positive per anchor (jaccard > 0.8);
  * anchors partitioned into EQUAL lexical-similarity bins so MRR/Hits can be
    reported per bin (the RQ1 evidence curve: lexical decays as cos->0, the
    result-aware method stays flat);
  * 20 HARD negatives per anchor: lexically nearest queries that are NOT the
    anchor's result-equivalent positives (mined from the full triplets corpus).

Lexical similarity is TF-IDF cosine over RAW SPARQL text (word 1-2 grams),
identical to evaluate_rq1._lexical_cos_fn, so the bins mean exactly what they
mean in the paper's stratified analysis.

Bins (equal size, scarce-bin-first greedy assignment):
    <0.3      adversarial / surface-divergent
    0.3-0.5   low overlap
    0.5-0.8   partial overlap
    >=0.8     near-duplicate (kept here as the "easy" contrast bin)

Hard-negative definition
------------------------
A hard negative for anchor a is a query q from the full triplets corpus with high
TF-IDF cos(a, q) that is NOT one of a's result-equivalent positives and (where a
jaccard label exists for the pair) has jaccard <= 0.8. Lexically similar, not
result-equivalent -> forces result/structure signal to reject.

Output: datasets/wikidata_testset_RQ1_stratified.json  (source schema + per-entry
lexical_bin / lexical_cos / answer_jaccard; negatives tagged neg_type="hard").

    python scripts/build_stratified_testset_wikidata.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "datasets" / "wikidata_testset_RQ1.json"
SRC_PATH = ROOT / "datasets" / "wikidata-triplets-dataset.json"
OUT_PATH = ROOT / "datasets" / "wikidata_testset_RQ1_stratified.json"

# (name, lo, hi) — equal target fraction across all four
BINS = [
    ("<0.3", 0.00, 0.30),
    ("0.3-0.5", 0.30, 0.50),
    ("0.5-0.8", 0.50, 0.80),
    (">=0.8", 0.80, 1.0001),
]
JACCARD_MIN = 0.8       # positive must be result-equivalent
N_HARD_NEG = 20         # hard negatives per anchor
SEED = 42

_norm = lambda s: " ".join(s.split())
_BIN_ORDER = {b[0]: i for i, b in enumerate(BINS)}
_bin_names = [b[0] for b in BINS]


def bin_of(c: float):
    for name, lo, hi in BINS:
        if lo <= c < hi:
            return name
    return None


def load_jaccard_lookup():
    """LUT[(anchor, query)] = jaccard, and the distinct query-text corpus."""
    src = json.load(open(SRC_PATH))
    lut, pool = {}, set()
    for g in src:
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
                    lut[(a, qn)] = float(p["answer_jaccard"])
    return lut, sorted(pool)


def main() -> None:
    rng = np.random.default_rng(SEED)
    test = json.load(open(TEST_PATH))
    jac, corpus_pool = load_jaccard_lookup()

    # --- Binning TF-IDF: fit on the TEST anchors + positives ONLY, so the
    # cos<0.3 boundary means exactly what it means for the balanced split and
    # evaluate_rq1._lexical_cos_fn (bins are directly comparable across splits). ---
    btexts = set()
    for e in test:
        btexts.add(_norm(e["anchor_query"]))
        for p in e["positive_queries"]:
            btexts.add(_norm(p))
    btexts = list(btexts)
    bidx = {q: i for i, q in enumerate(btexts)}
    Xb = sk_normalize(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2)).fit_transform(btexts),
        norm="l2", axis=1,
    )
    cos = lambda a, p: float(Xb[bidx[a]].multiply(Xb[bidx[p]]).sum())

    # --- Mining TF-IDF: separate space over the full triplets corpus, used only
    # to rank hard-negative candidates by lexical similarity to the anchor. ---
    ridx = {q: i for i, q in enumerate(corpus_pool)}
    X = sk_normalize(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2)).fit_transform(corpus_pool),
        norm="l2", axis=1,
    )

    # --- candidate result-equivalent positives per anchor, binned ---
    cand = defaultdict(lambda: defaultdict(list))   # cand[anchor][bin] = [(pos_raw, cos, jac)]
    entry_by_anchor = {}
    for e in test:
        a_raw = e["anchor_query"]
        a = _norm(a_raw)
        entry_by_anchor[a] = e
        for p_raw in e["positive_queries"]:
            p = _norm(p_raw)
            j = jac.get((a, p))
            if j is None or j <= JACCARD_MIN:
                continue
            b = bin_of(cos(a, p))
            if b is not None:
                cand[a][b].append((p_raw, cos(a, p), j))

    eligible = {a: set(bs) for a, bs in cand.items() if bs}
    avail = {b: sum(b in s for s in eligible.values()) for b in _bin_names}
    print(f"eligible anchors (>=1 result-equiv positive): {len(eligible)}")
    print("available per bin:", avail)

    # --- equal-size assignment, scarce-bin-first; max N with all bins >= N/4 ---
    def assign(need):
        got = {b: 0 for b in _bin_names}
        ch = {}
        order = sorted(eligible, key=lambda a: (len(eligible[a]), rng.random()))
        for a in order:
            opts = [b for b in _bin_names if b in eligible[a] and got[b] < need[b]]
            if opts:
                b = max(opts, key=lambda b: need[b] - got[b])
                got[b] += 1
                ch[a] = b
        return got, ch

    chosen = None
    for per_bin in range(min(avail.values()), 0, -1):
        need = {b: per_bin for b in _bin_names}
        got, ch = assign(need)
        if all(got[b] >= need[b] for b in need):
            chosen = ch
            print(f"equal-bin split: {per_bin}/bin  (N={per_bin * len(_bin_names)})")
            break

    def pick(bin_name, cands):
        lo, hi = next((b[1], b[2]) for b in BINS if b[0] == bin_name)
        if bin_name == _bin_names[0]:
            return min(cands, key=lambda t: t[1])           # hardest in lowest bin
        centre = (lo + hi) / 2.0
        return min(cands, key=lambda t: abs(t[1] - centre)) # nearest bin centre

    # --- hard-negative mining: X rows already ARE the pool (ridx maps pool) ---
    out, counts = [], defaultdict(int)
    for a, b in chosen.items():
        e = entry_by_anchor[a]
        p_raw, c, j = pick(b, cand[a][b])

        # exclude the anchor's own result-equivalent positives (all bins) + self
        pos_texts = {_norm(pp) for bb in cand[a] for (pp, _, _) in cand[a][bb]}
        pos_texts.add(a)

        sims = (X @ X[ridx[a]].T).toarray().ravel()
        order = np.argsort(-sims)
        hard = []
        for i in order:
            q = corpus_pool[i]
            if q in pos_texts:
                continue
            if jac.get((a, q), 0.0) > JACCARD_MIN:          # belt-and-suspenders
                continue
            hard.append({"query": q, "answer_jaccard": jac.get((a, q)),
                         "neg_type": "hard", "lexical_cos": round(float(sims[i]), 4)})
            if len(hard) >= N_HARD_NEG:
                break

        out.append({
            "anchor_query": e["anchor_query"],
            "positive_queries": [p_raw],
            "negative_queries": hard,
            "lexical_bin": b,
            "lexical_cos": round(c, 4),
            "answer_jaccard": round(j, 4),
        })
        counts[b] += 1

    out.sort(key=lambda r: (_BIN_ORDER[r["lexical_bin"]], r["lexical_cos"]))
    json.dump(out, open(OUT_PATH, "w"), ensure_ascii=False, indent=1)

    print(f"\nwrote {len(out)} entries to {OUT_PATH}")
    for b in _bin_names:
        cs = [r["lexical_cos"] for r in out if r["lexical_bin"] == b]
        print(f"  {b:>8}: n={counts[b]:>3}  cos[min={min(cs):.3f} "
              f"mean={np.mean(cs):.3f} max={max(cs):.3f}]")
    neg_cos = [n["lexical_cos"] for r in out for n in r["negative_queries"]]
    nneg = [len(r["negative_queries"]) for r in out]
    print(f"  hard negatives/anchor: min={min(nneg)} mean={np.mean(nneg):.1f} max={max(nneg)}")
    print(f"  hard-neg lexical cos to anchor: mean={np.mean(neg_cos):.3f} "
          f"median={np.median(neg_cos):.3f}")


if __name__ == "__main__":
    main()
