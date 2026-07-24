#!/usr/bin/env python3
"""Build a lexically-balanced, result-equivalent Wikidata RQ1 test split.

Motivation
----------
The standard RQ1 test set (datasets/wikidata_testset_RQ1.json) is dominated by
lexically-similar anchor/positive pairs, so an aggregate score largely measures
the easy, near-duplicate regime. This split re-balances the SAME result-
equivalent pairs so each lexical regime carries enough anchors to report per-bin
metrics, and drops the trivial near-duplicate regime entirely.

Every retained pair is result-equivalent (answer_jaccard > 0.8) — in fact every
positive in the source test set already satisfies this (min observed 0.802), so
result-equivalence is guaranteed, not merely filtered.

Design (current): two bins, 50/50, near-duplicates removed
----------------------------------------------------------
    low : cos < 0.30          -- surface-divergent / adversarial (50%)
    mid : 0.30 <= cos < 0.80  -- partial overlap (50%)
    (cos >= 0.80  DROPPED)    -- near-duplicate; trivially retrievable by any
                                 text method, so excluded to keep the benchmark
                                 focused on the regimes that discriminate methods.

Lexical similarity is TF-IDF cosine over the RAW SPARQL text (word 1-2 grams),
identical to evaluate_rq1._lexical_cos_fn. Raw (not label-expanded) text is used
so the bins mean exactly what they mean in the paper's stratified analysis.

Threshold rationale
-------------------
- 0.30: the boundary already established in the paper's stratified analysis
  (the <0.3 bin); ~ the median of the pair distribution (p50 = 0.288). Below it,
  anchor and its result-equivalent positive share almost no vocabulary/bigrams,
  so lexical and text-embedding signals are near-random and only result-aware /
  structural matching can retrieve.
- 0.80: near-duplicate boundary. At/above it the anchor and positive are almost
  identical in surface form; retrieval is trivial for every method, so these
  pairs add no discriminative signal and are removed.

One positive per anchor (so an anchor never appears twice and its other
positives never become false distractors in the shared corpus). Anchors are
greedily assigned scarce-bin-first to hit 50/50.

Output: datasets/wikidata_testset_RQ1_balanced.json, same schema as the source
plus per-entry lexical_bin / lexical_cos / answer_jaccard.

    python scripts/build_balanced_testset_wikidata.py
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
OUT_PATH = ROOT / "datasets" / "wikidata_testset_RQ1_balanced.json"

# Bins as (name, lo, hi, target_fraction); pairs with cos >= DROP_ABOVE removed.
# target_fraction is used only when BALANCE=True (subsample to those proportions).
# When BALANCE=False, every eligible anchor is kept and tagged by its sub-bin.
BINS = [
    ("low", 0.00, 0.30, 0.50),
    ("mid", 0.30, 0.50, 0.50),
]
DROP_ABOVE = 0.50      # keep only the surface-divergent regime (cos < 0.5)
BALANCE = False        # False: include ALL eligible anchors (no subsampling)
JACCARD_MIN = 0.8
SEED = 42

_norm = lambda s: " ".join(s.split())
_BIN_ORDER = {b[0]: i for i, b in enumerate(BINS)}


def bin_of(c: float):
    """Bin name for cosine c, or None if it is dropped (>= DROP_ABOVE / out of range)."""
    if c >= DROP_ABOVE:
        return None
    for name, lo, hi, _ in BINS:
        if lo <= c < hi:
            return name
    return None


def load_jaccard_lookup() -> dict:
    src = json.load(open(SRC_PATH))
    lut = {}
    for g in src:
        a = _norm(g["anchor_query"])
        for p in g.get("positive_queries", []):
            if p.get("query") is not None and p.get("answer_jaccard") is not None:
                lut[(a, _norm(p["query"]))] = float(p["answer_jaccard"])
    return lut


def main() -> None:
    rng = np.random.default_rng(SEED)
    test = json.load(open(TEST_PATH))
    jac = load_jaccard_lookup()

    texts = set()
    for e in test:
        texts.add(_norm(e["anchor_query"]))
        for p in e["positive_queries"]:
            texts.add(_norm(p))
    texts = list(texts)
    ridx = {q: i for i, q in enumerate(texts)}
    X = sk_normalize(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2)).fit_transform(texts),
        norm="l2", axis=1,
    )
    cos = lambda a, p: float(X[ridx[a]].multiply(X[ridx[p]]).sum())

    # cand[anchor][bin] = list of (original_positive, cos, jaccard)
    cand = defaultdict(lambda: defaultdict(list))
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

    eligible = {a: set(bins) for a, bins in cand.items() if bins}
    bin_names = [b[0] for b in BINS]
    print(f"anchors with >=1 result-equivalent, in-range positive: {len(eligible)}")
    print("eligible per bin:", {b: sum(b in s for s in eligible.values()) for b in bin_names})

    if not BALANCE:
        # Include every eligible anchor, tagged by the bin of its hardest
        # (lowest-cos) qualifying positive across the whole in-range set.
        chosen = {}
        for a, bins in cand.items():
            all_c = [(c, b) for b in bins for (_, c, _) in cand[a][b]]
            _, b = min(all_c, key=lambda t: t[0])
            chosen[a] = b
        print(f"include-all split (cos < {DROP_ABOVE}): N={len(chosen)}")
    else:
        target = {b[0]: b[3] for b in BINS}

        def assign(need):
            got = {b: 0 for b in bin_names}
            ch = {}
            order = sorted(eligible, key=lambda a: (len(eligible[a]), rng.random()))
            for a in order:
                opts = [b for b in bin_names if b in eligible[a] and got[b] < need[b]]
                if opts:
                    b = max(opts, key=lambda b: need[b] - got[b])
                    got[b] += 1
                    ch[a] = b
            return got, ch

        best = None
        for N in range(len(eligible), 0, -1):
            need = {b: round(target[b] * N) for b in bin_names}
            need[bin_names[-1]] += N - sum(need.values())
            got, ch = assign(need)
            if all(got[b] >= need[b] for b in need):
                best = (N, need, ch)
                break
        N, need, chosen = best
        print(f"balanced split: N={N}  target={need}")

    def pick(bin_name, cands):
        lo, hi = next((b[1], b[2]) for b in BINS if b[0] == bin_name)
        if bin_name == bin_names[0]:
            return min(cands, key=lambda t: t[1])          # hardest in lowest bin
        centre = (lo + hi) / 2.0
        return min(cands, key=lambda t: abs(t[1] - centre))

    out, counts = [], defaultdict(int)
    for a, b in chosen.items():
        e = entry_by_anchor[a]
        p_raw, c, j = pick(b, cand[a][b])
        out.append({
            "anchor_query": e["anchor_query"],
            "positive_queries": [p_raw],
            "negative_queries": e.get("negative_queries", []),
            "lexical_bin": b,
            "lexical_cos": round(c, 4),
            "answer_jaccard": round(j, 4),
        })
        counts[b] += 1

    out.sort(key=lambda r: (_BIN_ORDER[r["lexical_bin"]], r["lexical_cos"]))
    json.dump(out, open(OUT_PATH, "w"), ensure_ascii=False, indent=1)

    print(f"wrote {len(out)} entries to {OUT_PATH}")
    for b in bin_names:
        cs = [r["lexical_cos"] for r in out if r["lexical_bin"] == b]
        print(f"  {b:>4}: n={counts[b]:>3} ({counts[b]/len(out):.0%})  "
              f"cos[min={min(cs):.3f} mean={np.mean(cs):.3f} max={max(cs):.3f}]")
    js = [r["answer_jaccard"] for r in out]
    print(f"  answer_jaccard: min={min(js):.3f} (all > {JACCARD_MIN}); dropped cos >= {DROP_ABOVE}")


if __name__ == "__main__":
    main()
