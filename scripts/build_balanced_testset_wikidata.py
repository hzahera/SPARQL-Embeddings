#!/usr/bin/env python3
"""Build a lexically-balanced, result-equivalent Wikidata RQ1 test split.

Motivation
----------
The standard RQ1 test set (datasets/wikidata_testset_RQ1.json) is dominated by
lexically-similar anchor/positive pairs, so an aggregate score is largely a
measure of the easy, near-duplicate regime. This split re-balances the SAME
result-equivalent pairs across three lexical-similarity regimes so every regime
carries enough anchors to report per-bin MRR/Hits/nDCG, and so the aggregate is
not swamped by easy cases.

Every retained pair is result-equivalent (answer_jaccard > 0.8) — in fact every
positive in the source test set already satisfies this (min observed 0.802), so
result-equivalence is guaranteed, not merely filtered.

Lexical similarity
------------------
Anchor--positive similarity is TF-IDF cosine over the RAW SPARQL text (word 1-2
grams), identical to evaluate_rq1._lexical_cos_fn. Raw (not label-expanded) text
is used so the bins mean exactly what they mean in the paper's existing
stratified analysis (the <0.3 bin), keeping this split comparable to it.

Bins (partition of [0, 1]):
    low    : cos < 0.30   -- surface-divergent / adversarial. Anchor and its
                             result-equivalent positive share almost no
                             vocabulary or bigrams; lexical (TF-IDF/BM25) and
                             text-embedding signals are near-random here, so only
                             result-aware / structural matching can retrieve.
                             Reuses the 0.30 boundary already established in the
                             paper's stratified analysis; 0.30 ~ the median of
                             the pair distribution (p50 = 0.288).
    medium : 0.30 <= cos < 0.60 -- partial overlap; the transition regime where
                             methods begin to differentiate.
    high   : cos >= 0.60   -- near-paraphrase (~p80). Dominant shared surface
                             form; text encoders retrieve these trivially. Kept
                             as a smaller "easy" control regime that confirms the
                             benchmark is not degenerate (all methods, including
                             text baselines, should score well here).

Target composition: 40% low, 40% medium, 20% high. Low+medium (80%) deliberately
over-represents the regimes that stress result-aware retrieval relative to their
difficulty, while the 20% high control keeps a sanity check that surface form,
when present, is still exploited.

Construction
-----------
One positive per anchor (so an anchor never appears twice and its other
positives never become false distractors in the shared corpus). Anchors are
greedily assigned to bins scarce-bin-first to hit the target proportions. The
representative positive per (anchor, bin) is the hardest/most-central one:
lowest cos for `low`, highest for `high`, nearest the bin centre for `medium`.

Output: datasets/wikidata_testset_RQ1_balanced.json, same schema as the source
(list of {anchor_query, positive_queries, negative_queries}) plus per-entry
metadata (lexical_bin, lexical_cos, answer_jaccard) that the evaluation pipeline
ignores but which lets you filter by bin.

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

LOW_T, HIGH_T = 0.30, 0.60          # bin boundaries (see module docstring)
JACCARD_MIN = 0.8                   # result-equivalence threshold
TARGET = {"low": 0.40, "medium": 0.40, "high": 0.20}
SEED = 42

_norm = lambda s: " ".join(s.split())


def bin_of(c: float) -> str:
    return "low" if c < LOW_T else ("medium" if c < HIGH_T else "high")


def load_jaccard_lookup() -> dict:
    """(anchor, positive) -> answer_jaccard from the source triplet dataset."""
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

    # Fit TF-IDF exactly as evaluate_rq1._lexical_cos_fn (raw text, word 1-2 grams).
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

    # Per anchor: candidate positives per bin, each result-equivalent (>0.8).
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
            if j is None or j <= JACCARD_MIN:      # keep only result-equivalent
                continue
            c = cos(a, p)
            cand[a][bin_of(c)].append((p_raw, c, j))

    eligible = {a: set(bins) for a, bins in cand.items()}
    n_total = len(eligible)
    per_bin_elig = {b: sum(b in s for s in eligible.values()) for b in ("low", "medium", "high")}
    print(f"anchors with >=1 result-equivalent positive: {n_total}")
    print(f"eligible per bin: {per_bin_elig}")

    # Largest N whose 40/40/20 need is satisfiable by greedy scarce-first assign.
    def assign(need):
        got = {"low": 0, "medium": 0, "high": 0}
        chosen = {}
        # anchors with the fewest options first; break ties randomly (seeded)
        order = sorted(eligible, key=lambda a: (len(eligible[a]), rng.random()))
        for a in order:
            opts = [b for b in ("low", "medium", "high")
                    if b in eligible[a] and got[b] < need[b]]
            if not opts:
                continue
            b = max(opts, key=lambda b: need[b] - got[b])   # neediest bin first
            got[b] += 1
            chosen[a] = b
        return got, chosen

    best = None
    for N in range(n_total, 0, -1):
        need = {"low": round(TARGET["low"] * N),
                "medium": round(TARGET["medium"] * N)}
        need["high"] = N - need["low"] - need["medium"]
        got, chosen = assign(need)
        if all(got[b] >= need[b] for b in need):
            best = (N, need, chosen)
            break
    N, need, chosen = best
    print(f"balanced split: N={N}  target={need}")

    # Pick the representative positive per (anchor, assigned bin).
    def pick(bin_name, cands):
        if bin_name == "low":
            return min(cands, key=lambda t: t[1])          # hardest (lowest cos)
        if bin_name == "high":
            return max(cands, key=lambda t: t[1])          # easiest (highest cos)
        centre = (LOW_T + HIGH_T) / 2.0
        return min(cands, key=lambda t: abs(t[1] - centre))  # most central

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

    # Stable, reproducible order: by bin then cos.
    out.sort(key=lambda r: ({"low": 0, "medium": 1, "high": 2}[r["lexical_bin"]], r["lexical_cos"]))
    json.dump(out, open(OUT_PATH, "w"), ensure_ascii=False, indent=1)

    print(f"wrote {len(out)} entries to {OUT_PATH}")
    for b in ("low", "medium", "high"):
        cs = [r["lexical_cos"] for r in out if r["lexical_bin"] == b]
        print(f"  {b:>6}: n={counts[b]:>3} ({counts[b]/len(out):.0%})  "
              f"cos[min={min(cs):.3f} mean={np.mean(cs):.3f} max={max(cs):.3f}]")
    js = [r["answer_jaccard"] for r in out]
    print(f"  answer_jaccard: min={min(js):.3f} (all > {JACCARD_MIN})")


if __name__ == "__main__":
    main()
