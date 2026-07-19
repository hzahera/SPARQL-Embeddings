"""
Wikidata variant of build_hard_adversarial_rq1.py.

Wikidata's result_sets_wikidata.json (~6.4GB) cannot be loaded into memory
wholesale (individual query result sets run to the millions of rows -- a
2000-key sample averaged 17,362 rows/query, max 6.6M). Loading it or doing an
O(n_anchors x n_corpus) exact-Jaccard scan like the DBpedia script does is not
tractable. This script instead:

  Phase A (cheap, no big file): for each held-out anchor, find the top-K
    lexically-nearest candidates across the FULL Wikidata triplets corpus
    (TF-IDF cosine only) as hard-negative CANDIDATES, and take the anchor's
    own stored `positive_queries` (jaccard already known) as positive
    candidates.
  Phase B (one single streaming pass over the 6.4GB file): using ijson's
    low-level event parser, walk the file ONCE. For each top-level key,
    if it's in the "needed" set (anchors + candidate texts from phase A),
    accumulate its result rows (capped at --max-rows-per-query, default
    200,000, to bound memory/time for pathological huge answer sets --
    documented as an approximation for that handful of queries).
    Otherwise, skip the array's tokens without building any Python object.
  Phase C: compute exact (or capped-approximate) Jaccard for only the
    (anchor, candidate) pairs identified in phase A, then rank/pad to
    --target-n exactly like the DBpedia script.

Run this on the cluster machine where result_sets_wikidata.json lives --
NOT in a memory-constrained sandbox. Expect a single-digit-hours streaming
pass depending on disk I/O; there is no way around one full linear read of
the file since keys of interest are scattered throughout it.

Usage:
  python build_hard_adversarial_rq1_wikidata.py \
      --triplets datasets/wikidata-triplets-dataset.json \
      --splits datasets/splits_wikidata.json \
      --result-sets datasets/result_sets_wikidata.json \
      --out datasets/wikidata_testset_RQ1_hard.json \
      --target-n 500 --top-k-candidates 30
"""
import argparse
import json
import re
import sys

import ijson
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip()


def row_values(obj: dict) -> tuple:
    return tuple(sorted(v["value"] for v in obj.values()))


def jacc(a_set, b_set) -> float:
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def stream_needed_result_sets(path, needed_keys, max_rows_per_query=200_000):
    """Single pass over the huge JSON object, keeping only `needed_keys`.
    Returns {normalized_key: set(row_value_tuples)}."""
    out = {}
    parser = ijson.parse(open(path, "rb"), use_float=True)
    depth = 0
    cur_key = None
    cur_norm_key = None
    want = False
    cur_rows = None
    n_seen = 0
    n_kept = 0
    for prefix, event, value in parser:
        if event == "map_key" and depth == 0:
            cur_key = value
            cur_norm_key = norm(cur_key)
            want = cur_norm_key in needed_keys
            if want:
                cur_rows = []
            n_seen += 1
            if n_seen % 200000 == 0:
                print(f"  ...scanned {n_seen} keys, kept {n_kept}", file=sys.stderr)
        elif event == "start_array":
            depth += 1
        elif event == "end_array":
            depth -= 1
            if depth == 0 and want:
                out[cur_norm_key] = set(cur_rows)
                n_kept += 1
                cur_rows = None
                want = False
        elif event == "string" and want and depth == 1:
            if len(cur_rows) < max_rows_per_query:
                try:
                    cur_rows.append(row_values(json.loads(value)))
                except Exception:
                    pass
    print(f"streamed: saw {n_seen} keys, kept {n_kept}/{len(needed_keys)} needed", file=sys.stderr)
    return out


def build(triplets_path, splits_path, result_sets_path, out_path,
          pos_jacc_min=0.8, pos_cos_max=0.3,
          neg_jacc_max=0.2, neg_cos_min=0.8,
          target_n=500, top_k_per_anchor=3, top_k_candidates=30,
          max_rows_per_query=200_000):

    d = json.load(open(triplets_path))
    splits = json.load(open(splits_path))
    test_idx = list(splits["test_idx"])

    # ---- Phase A: cheap lexical candidate search over the FULL corpus ----
    print("Phase A: building corpus + TF-IDF ...", file=sys.stderr)
    all_texts = set()
    for it in d:
        all_texts.add(norm(it["anchor_query"]))
        for p in it.get("positive_queries", []):
            all_texts.add(norm(p["query"]))
        for n in it.get("negative_queries", []):
            all_texts.add(norm(n["query"]))
    corpus_texts = list(all_texts)
    vec = TfidfVectorizer().fit(corpus_texts)
    X = vec.transform(corpus_texts)
    text2idx = {t: i for i, t in enumerate(corpus_texts)}

    anchor_local = {}  # idx -> dict with local pos/neg candidates + top-K global neg candidates (text only)
    needed_keys = set()

    for i in test_idx:
        it = d[i]
        a = norm(it["anchor_query"])
        needed_keys.add(a)
        local_pos = [(norm(p["query"]), p.get("answer_jaccard", 0.0)) for p in it.get("positive_queries", [])]
        local_neg = [(norm(n["query"]), n.get("answer_jaccard", 1.0)) for n in it.get("negative_queries", [])]
        for t, _ in local_pos + local_neg:
            needed_keys.add(t)

        # top-K global lexical neighbors as hard-negative candidates
        global_neg_candidates = []
        if a in text2idx:
            ai = text2idx[a]
            sims = cosine_similarity(X[ai], X)[0]
            order = np.argsort(-sims)
            count = 0
            for j in order:
                if j == ai:
                    continue
                t = corpus_texts[j]
                global_neg_candidates.append(t)
                needed_keys.add(t)
                count += 1
                if count >= top_k_candidates:
                    break

        anchor_local[i] = {"anchor": a, "local_pos": local_pos, "local_neg": local_neg,
                           "global_neg_candidates": global_neg_candidates}

    print(f"Phase A done: {len(test_idx)} anchors, {len(needed_keys)} distinct query strings to resolve", file=sys.stderr)

    # ---- Phase B: single streaming pass over the huge result-sets file ----
    print("Phase B: streaming result_sets_wikidata.json (this is the slow part) ...", file=sys.stderr)
    rs_sets = stream_needed_result_sets(result_sets_path, needed_keys, max_rows_per_query)

    # ---- Phase C: rank + assemble, mirrors build_hard_adversarial_rq1.py ----
    print("Phase C: scoring candidates ...", file=sys.stderr)

    def cos_of(a_text, b_text):
        if a_text not in text2idx or b_text not in text2idx:
            return None
        return float(cosine_similarity(X[text2idx[a_text]], X[text2idx[b_text]])[0][0])

    def meets_bar(p, n):
        return bool(p[1] >= pos_jacc_min and p[2] <= pos_cos_max and
                    n[1] <= neg_jacc_max and n[2] >= neg_cos_min)

    anchor_results = []
    for i, info in anchor_local.items():
        a = info["anchor"]
        pos_candidates = []
        for t, stored_j in info["local_pos"]:
            c = cos_of(a, t)
            if c is not None:
                pos_candidates.append((t, stored_j, c))
        neg_candidates = []
        for t, stored_j in info["local_neg"]:
            c = cos_of(a, t)
            if c is not None:
                neg_candidates.append((t, stored_j, c))
        # global lexical candidates verified with exact/capped Jaccard from streamed sets
        if a in rs_sets:
            a_set = rs_sets[a]
            for t in info["global_neg_candidates"]:
                if t in rs_sets:
                    j = jacc(a_set, rs_sets[t])
                    c = cos_of(a, t)
                    if c is not None:
                        neg_candidates.append((t, j, c))
                        pos_candidates.append((t, j, c))

        if not pos_candidates or not neg_candidates:
            continue

        pos_sorted = sorted([c for c in pos_candidates if c[1] >= pos_jacc_min], key=lambda c: c[2])
        if not pos_sorted:
            pos_sorted = sorted(pos_candidates, key=lambda c: (-c[1], c[2]))[:top_k_per_anchor]
        neg_sorted = sorted([c for c in neg_candidates if c[1] <= neg_jacc_max], key=lambda c: -c[2])
        if not neg_sorted:
            neg_sorted = sorted(neg_candidates, key=lambda c: (c[1], -c[2]))[:top_k_per_anchor]

        pairs = []
        for p in pos_sorted[:top_k_per_anchor]:
            for n in neg_sorted[:top_k_per_anchor]:
                score = (p[1] - p[2]) + (n[2] - n[1])
                pairs.append((p, n, score))
        pairs.sort(key=lambda x: -x[2])
        anchor_results.append((i, a, pairs[:top_k_per_anchor]))

    anchor_results.sort(key=lambda ar: -ar[2][0][2] if ar[2] else 0)

    rows = []
    for i, a, pairs in anchor_results:
        if not pairs:
            continue
        p, n, score = pairs[0]
        rows.append({
            "anchor_query": a,
            "positive_query": p[0], "positive_jaccard": round(p[1], 4), "positive_cosine": round(p[2], 4),
            "negative_query": n[0], "negative_jaccard": round(n[1], 4), "negative_cosine": round(n[2], 4),
            "hardness_score": round(score, 4), "meets_strict_bar": meets_bar(p, n), "slot": 0,
        })

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
                    "positive_query": p[0], "positive_jaccard": round(p[1], 4), "positive_cosine": round(p[2], 4),
                    "negative_query": n[0], "negative_jaccard": round(n[1], 4), "negative_cosine": round(n[2], 4),
                    "hardness_score": round(score, 4), "meets_strict_bar": meets_bar(p, n), "slot": k,
                })
                slot_idx[i] = k
                added = True
        if not added:
            break

    rows.sort(key=lambda r: -r["hardness_score"])
    rows = rows[:target_n]

    n_unique = len(set(r["anchor_query"] for r in rows))
    n_strict = sum(1 for r in rows if r["meets_strict_bar"])
    print(f"wikidata: {len(rows)} rows from {n_unique} unique anchors (of {len(test_idx)} total test anchors)")
    print(f"  {n_strict}/{len(rows)} rows meet the FULL strict bar")

    json.dump(rows, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--result-sets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-n", type=int, default=500)
    ap.add_argument("--top-k-candidates", type=int, default=30,
                     help="how many lexically-nearest global candidates to verify per anchor for hard negatives")
    ap.add_argument("--max-rows-per-query", type=int, default=200_000,
                     help="cap on rows collected per query; queries with larger result sets get an approximate (truncated) Jaccard")
    ap.add_argument("--pos-jaccard-min", type=float, default=0.8)
    ap.add_argument("--pos-cosine-max", type=float, default=0.3)
    ap.add_argument("--neg-jaccard-max", type=float, default=0.2)
    ap.add_argument("--neg-cosine-min", type=float, default=0.8)
    args = ap.parse_args()

    build(args.triplets, args.splits, args.result_sets, args.out,
          pos_jacc_min=args.pos_jaccard_min, pos_cos_max=args.pos_cosine_max,
          neg_jacc_max=args.neg_jaccard_max, neg_cos_min=args.neg_cosine_min,
          target_n=args.target_n, top_k_candidates=args.top_k_candidates,
          max_rows_per_query=args.max_rows_per_query)
