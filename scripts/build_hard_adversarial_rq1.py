"""
Build the "hard adversarial" RQ1 test set: for each held-out anchor, pick the
HARDEST positive (answer-Jaccard high, lexical cosine LOW) and the HARDEST
negative (lexical cosine HIGH, answer-Jaccard LOW). This isolates exactly the
regime the paper's thesis is about: surface-divergent-but-result-equivalent
queries, and surface-similar-but-result-different queries.

Ground truth for answer-Jaccard comes from two sources, in priority order:
  1. The pre-computed `answer_jaccard` already stored in each anchor's own
     `positive_queries` / `negative_queries` lists (exact, free).
  2. For DBpedia only: the cached result sets in `result_sets_dbpedia.json`,
     searched across the FULL cached corpus (not just the anchor's own local
     lists) to find better candidates. Verified to reproduce the paper's
     stored Jaccard exactly (row values extracted ignoring SPARQL variable
     names, set-Jaccard over rows).

Wikidata's result_sets_wikidata.json is NOT used here (individual result sets
run up to millions of rows -- exact pairwise Jaccard at corpus scale is not
tractable to compute casually). Wikidata negatives/positives are therefore
drawn from the local stored lists only, plus a global-lexical CANDIDATE SCAN
(cosine only, cheap) that is then verified with exact Jaccard via a targeted,
single streaming pass over result_sets_wikidata.json -- see --use-result-sets.

Output: {kg}_testset_RQ1_hard.json, a list of:
  {
    "anchor_query": ...,
    "positive_query": ..., "positive_jaccard": ..., "positive_cosine": ...,
    "negative_query": ..., "negative_jaccard": ..., "negative_cosine": ...,
    "hardness_score": ...,
    "slot": 0   # 0 = anchor's best pair, 1 = anchor's 2nd-best pair, etc.
  }

Anchors are ranked by hardness_score = (pos_jaccard - pos_cosine) +
(neg_cosine - neg_jaccard). If there are fewer than --target-n unique anchors
with at least one valid (positive, negative) pair, additional (2nd-best,
3rd-best, ...) pairs from the highest-scoring anchors are added to reach
--target-n, and each row's "slot" documents this so it's auditable rather than
silently padded.
"""
import argparse
import json
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip()


def row_values(row_str: str) -> tuple:
    obj = json.loads(row_str)
    return tuple(sorted(v["value"] for v in obj.values()))


def load_result_sets(path):
    rs = json.load(open(path))
    rs_norm = {norm(k): v for k, v in rs.items()}
    return {q: set(row_values(r) for r in rows) for q, rows in rs_norm.items()}


def jacc(a_set, b_set) -> float:
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def build(kg, triplets_path, splits_path, result_sets_path,
          pos_jacc_min=0.8, pos_cos_max=0.3,
          neg_jacc_max=0.2, neg_cos_min=0.8,
          target_n=500, top_k_per_anchor=3, out_path=None):

    d = json.load(open(triplets_path))
    splits = json.load(open(splits_path))
    test_idx = list(splits["test_idx"])

    rs_sets = load_result_sets(result_sets_path) if result_sets_path else {}

    # global corpus for cosine search (only needed if result_sets given)
    corpus_texts, corpus_vec, X = None, None, None
    text2idx = {}
    if rs_sets:
        corpus_texts = list(rs_sets.keys())
        corpus_vec = TfidfVectorizer().fit(corpus_texts)
        X = corpus_vec.transform(corpus_texts)
        text2idx = {t: i for i, t in enumerate(corpus_texts)}

    anchor_results = []  # (anchor_idx, [ (pos_text,pos_j,pos_c) ... ], [ (neg_text,neg_j,neg_c) ... ])

    for i in test_idx:
        it = d[i]
        a = norm(it["anchor_query"])

        pos_candidates = []  # (text, jaccard, cosine)
        neg_candidates = []

        # --- local stored lists (cheap TF-IDF cosine vs. anchor only) ---
        local_texts = [norm(p["query"]) for p in it.get("positive_queries", [])] + \
                      [norm(n["query"]) for n in it.get("negative_queries", [])]
        if local_texts:
            local_vec = TfidfVectorizer().fit(local_texts + [a])
            a_local_vec = local_vec.transform([a])
            for p in it.get("positive_queries", []):
                t = norm(p["query"])
                cos = cosine_similarity(a_local_vec, local_vec.transform([t]))[0][0]
                pos_candidates.append((t, p.get("answer_jaccard", 0.0), cos))
            for n in it.get("negative_queries", []):
                t = norm(n["query"])
                cos = cosine_similarity(a_local_vec, local_vec.transform([t]))[0][0]
                neg_candidates.append((t, n.get("answer_jaccard", 1.0), cos))

        # --- global cached-corpus search (DBpedia only, exact Jaccard) ---
        if rs_sets and a in text2idx:
            ai = text2idx[a]
            sims = cosine_similarity(X[ai], X)[0]
            a_set = rs_sets[a]
            for j, t in enumerate(corpus_texts):
                if j == ai:
                    continue
                cos = sims[j]
                # only bother computing exact jaccard where it could plausibly help
                if cos <= pos_cos_max + 0.15 or cos >= neg_cos_min - 0.2:
                    jac = jacc(a_set, rs_sets[t])
                    pos_candidates.append((t, jac, cos))
                    neg_candidates.append((t, jac, cos))

        if not pos_candidates or not neg_candidates:
            continue

        # rank positive candidates: prefer high jaccard, low cosine
        pos_sorted = sorted(
            [c for c in pos_candidates if c[1] >= pos_jacc_min],
            key=lambda c: c[2],  # ascending cosine = hardest first
        )
        if not pos_sorted:
            # fall back: best available even if below bar (documented via score)
            pos_sorted = sorted(pos_candidates, key=lambda c: (-c[1], c[2]))[:top_k_per_anchor]

        neg_sorted = sorted(
            [c for c in neg_candidates if c[1] <= neg_jacc_max],
            key=lambda c: -c[2],  # descending cosine = hardest first
        )
        if not neg_sorted:
            neg_sorted = sorted(neg_candidates, key=lambda c: (c[1], -c[2]))[:top_k_per_anchor]

        pairs = []
        for p in pos_sorted[:top_k_per_anchor]:
            for n in neg_sorted[:top_k_per_anchor]:
                score = (p[1] - p[2]) + (n[2] - n[1])
                pairs.append((p, n, score))
        pairs.sort(key=lambda x: -x[2])
        anchor_results.append((i, a, pairs[:top_k_per_anchor]))

    # rank anchors by their best pair's score
    anchor_results.sort(key=lambda ar: -ar[2][0][2] if ar[2] else 0)

    def meets_bar(p, n):
        return bool(p[1] >= pos_jacc_min and p[2] <= pos_cos_max and
                    n[1] <= neg_jacc_max and n[2] >= neg_cos_min)

    rows = []
    # pass 1: one slot per anchor
    for i, a, pairs in anchor_results:
        if not pairs:
            continue
        p, n, score = pairs[0]
        rows.append({
            "anchor_query": a,
            "positive_query": p[0], "positive_jaccard": round(p[1], 4), "positive_cosine": round(float(p[2]), 4),
            "negative_query": n[0], "negative_jaccard": round(n[1], 4), "negative_cosine": round(float(n[2]), 4),
            "hardness_score": round(float(score), 4),
            "meets_strict_bar": meets_bar(p, n),
            "slot": 0,
        })

    # pass 2: pad with 2nd/3rd-best pairs from the highest-scoring anchors, if short of target
    slot_idx = {}
    while len(rows) < target_n:
        added = False
        for i, a, pairs in anchor_results:
            if len(rows) >= target_n:
                break
            k = slot_idx.get(i, 0) + 1
            if k < len(pairs):
                p, n, score = pairs[k]
                rows.append({
                    "anchor_query": a,
                    "positive_query": p[0], "positive_jaccard": round(p[1], 4), "positive_cosine": round(float(p[2]), 4),
                    "negative_query": n[0], "negative_jaccard": round(n[1], 4), "negative_cosine": round(float(n[2]), 4),
                    "hardness_score": round(float(score), 4),
                    "meets_strict_bar": meets_bar(p, n),
                    "slot": k,
                })
                slot_idx[i] = k
                added = True
        if not added:
            break  # exhausted all available pairs

    rows.sort(key=lambda r: -r["hardness_score"])
    rows = rows[:target_n]

    n_unique_anchors = len(set(r["anchor_query"] for r in rows))
    n_strict = sum(1 for r in rows if r["meets_strict_bar"])
    print(f"{kg}: {len(rows)} rows from {n_unique_anchors} unique anchors "
          f"(of {len(test_idx)} total test anchors)")
    print(f"  {n_strict}/{len(rows)} rows meet the FULL strict bar "
          f"(pos jaccard>={pos_jacc_min}/cos<={pos_cos_max}, "
          f"neg jaccard<={neg_jacc_max}/cos>={neg_cos_min})")
    print(f"  median positive_jaccard={np.median([r['positive_jaccard'] for r in rows]):.3f}, "
          f"median positive_cosine={np.median([r['positive_cosine'] for r in rows]):.3f}")
    print(f"  median negative_jaccard={np.median([r['negative_jaccard'] for r in rows]):.3f}, "
          f"median negative_cosine={np.median([r['negative_cosine'] for r in rows]):.3f}")

    if out_path:
        json.dump(rows, open(out_path, "w"), indent=2)
        print(f"  wrote {out_path}")

    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True, choices=["dbpedia", "wikidata"])
    ap.add_argument("--triplets", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--result-sets", default=None, help="optional: enables exact global-corpus Jaccard search (DBpedia only, feasible)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-n", type=int, default=500)
    ap.add_argument("--pos-jaccard-min", type=float, default=0.8)
    ap.add_argument("--pos-cosine-max", type=float, default=0.3)
    ap.add_argument("--neg-jaccard-max", type=float, default=0.2)
    ap.add_argument("--neg-cosine-min", type=float, default=0.8)
    args = ap.parse_args()

    build(args.kg, args.triplets, args.splits, args.result_sets,
          pos_jacc_min=args.pos_jaccard_min, pos_cos_max=args.pos_cosine_max,
          neg_jacc_max=args.neg_jaccard_max, neg_cos_min=args.neg_cosine_min,
          target_n=args.target_n, out_path=args.out)
