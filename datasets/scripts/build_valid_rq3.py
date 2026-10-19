#!/usr/bin/env python3
"""
Complete pair labels for the VALIDATION split, in the same format as <graph>-test-rq3.json,
built from the cached answer sets (no query is executed).

Every unordered pair of unique validation queries (anchors + stored candidates) is labelled:
  positive  J >= 0.8   -> positive_pairs [id_i, id_j, J]
  negative  J <= 0.1   -> negative_pairs [id_i, id_j, J]
  otherwise            -> counted as excluded_intermediate (not listed), as in the test file.
These labels are used only to select cosine thresholds on validation data (RQ3 calibration).

Run from the repository root:
  python datasets/scripts/build_valid_rq3.py --dataset dbpedia               # builds the validation file
  python datasets/scripts/build_valid_rq3.py --dataset dbpedia --split test  # check only: rebuilds the test
        labels the same way and compares the counts with the existing <graph>-test-rq3.json (nothing is written)
Output: datasets/<graph>-splits/<graph>-valid-rq3.json
"""
import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path


def normalize_query(q: str) -> str:          # identical to cache_eval / rebuild scripts
    return re.sub(r"\s+", " ", q.strip())


def query_hash(q: str) -> str:
    return hashlib.sha256(normalize_query(q).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dbpedia", "wikidata"])
    ap.add_argument("--positive-min", type=float, default=0.8)
    ap.add_argument("--negative-max", type=float, default=0.1)
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    args = ap.parse_args()
    g = args.dataset
    split_dir = Path(f"datasets/{g}-splits")
    src = split_dir / f"{g}-{args.split}.json"

    # workload = unique validation queries, exactly as cache_eval._global_unique_queries
    records = json.load(open(src))
    seen, queries = set(), []
    for r in records:
        for q in [r["anchor_query"]] + [c["query"] for c in r["positive_queries"] + r["negative_queries"]]:
            if q not in seen:
                seen.add(q)
                queries.append(q)
    hashes = [query_hash(q) for q in queries]
    if len(set(hashes)) != len(hashes):
        raise SystemExit("two validation queries share a normalized hash")
    need = set(hashes)
    print(f"{g}: {len(records)} {args.split} anchors, {len(queries)} unique {args.split} queries")

    # cached answers for these queries only
    t0 = time.perf_counter()
    answers = {}
    with open(split_dir / "query_execution_cache.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj["query_hash"] in need:
                if obj["status"] != "success":
                    raise SystemExit(f"validation query {obj['query_hash'][:12]} has status {obj['status']}")
                answers[obj["query_hash"]] = set(obj["answers"])
    missing = need - set(answers)
    if missing:
        raise SystemExit(f"{len(missing)} validation queries are not in the execution cache")
    sets = [answers[h] for h in hashes]
    print(f"loaded answers in {time.perf_counter() - t0:.0f}s")

    # intersections via an inverted index; pairs without a shared row have J = 0
    inv = defaultdict(list)
    for i, s in enumerate(sets):
        for row in s:
            inv[row].append(i)
    inter = defaultdict(int)
    for ids in inv.values():
        if len(ids) > 1:
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    inter[(ids[x], ids[y])] += 1

    n = len(queries)
    sizes = [len(s) for s in sets]
    pos, neg, middle = [], [], 0
    for i in range(n):
        for j in range(i + 1, n):
            k = inter.get((i, j), 0)
            jac = k / (sizes[i] + sizes[j] - k) if k else 0.0
            if jac >= args.positive_min:
                pos.append([i, j, round(jac, 12)])
            elif jac <= args.negative_max:
                neg.append([i, j, round(jac, 12)])
            else:
                middle += 1

    out = {
        "metadata": {
            "rq": "RQ3", "task": "semantic_cache_pair_ground_truth", "dataset": g, "split": "validation",
            "source_valid_file": str(src), "source": "cached answer sets (query_execution_cache.jsonl)",
            "query_count": n, "all_unordered_pair_count": n * (n - 1) // 2,
            "positive_jaccard_min": args.positive_min, "negative_jaccard_max": args.negative_max,
            "positive_pair_count": len(pos), "negative_pair_count": len(neg),
            "excluded_intermediate_pair_count": middle,
            "pair_coverage": "complete over all unordered pairs of unique validation queries",
            "omitted_pairs": "intermediate pairs are counted but not listed",
        },
        "queries": [{"id": i, "query_hash": hashes[i], "query": queries[i], "n_answers": sizes[i]} for i in range(n)],
        "positive_pairs": pos,
        "negative_pairs": neg,
    }
    if args.split == "test":
        ref = json.load(open(split_dir / f"{g}-test-rq3.json"))["metadata"]
        for k in ("query_count", "positive_pair_count", "negative_pair_count", "excluded_intermediate_pair_count"):
            mine = out["metadata"][k]
            print(f"  {k:34s} rebuilt {mine:>9}  existing {ref.get(k)!s:>9}  {'ok' if mine == ref.get(k) else 'DIFFERENT'}")
        return
    dest = split_dir / f"{g}-valid-rq3.json"
    json.dump(out, open(dest, "w"), ensure_ascii=False)
    print(json.dumps(out["metadata"], indent=1))
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()
