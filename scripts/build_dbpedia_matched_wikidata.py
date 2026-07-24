#!/usr/bin/env python3
"""Build a Wikidata RQ1 test split that MIRRORS the DBpedia test split.

Matches dbpedia_testset_RQ1.json on every construction parameter:
  * size:        462 anchors
  * positives:   jaccard > 0.8 (result-equivalent), capped at 20/anchor,
                 sampled to match DBpedia's positives-per-anchor histogram
                 (1:165  2-4:83  5-9:53  10-19:34  20:127  -> mean ~8.18)
  * negatives:   20/anchor, jaccard = 0 (Wikidata already ships 20 shared
                 negatives/anchor, like DBpedia's ~23)
  * thresholds:  positive jaccard > 0.8 ; negative jaccard = 0

Anchors are drawn from the held-out Wikidata TEST split only. Where a bucket is
short of the DBpedia target (only the 20-cap bucket, by 5), the deficit spills
into the next lower bucket so the total stays 462.

Output: datasets/wikidata_testset_RQ1_matched.json

    python scripts/build_dbpedia_matched_wikidata.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRIP = ROOT / "datasets" / "wikidata-triplets-dataset.json"
SPLITS = ROOT / "datasets" / "splits_wikidata.json"
OUT = ROOT / "datasets" / "wikidata_testset_RQ1_matched.json"

JAC_MIN = 0.80        # positive threshold (DBpedia min observed 0.804)
CAP = 20              # max positives / anchor (DBpedia cap)
SEED = 42

# DBpedia positives-per-anchor histogram (target), buckets high->low
BUCKETS = ["20", "10-19", "5-9", "2-4", "1"]
TARGET = {"1": 165, "2-4": 83, "5-9": 53, "10-19": 34, "20": 127}   # sums to 462


def bucket(c):
    return "1" if c == 1 else "2-4" if c <= 4 else "5-9" if c <= 9 \
        else "10-19" if c <= 19 else "20"


def main():
    rng = np.random.default_rng(SEED)
    trip = json.load(open(TRIP))
    spl = json.load(open(SPLITS))
    test = [trip[i] for i in spl["test_idx"]]

    # group anchors by #(jaccard>0.8) positives, capped
    by_bucket = defaultdict(list)
    for e in test:
        pos = [p for p in e["positive_queries"]
               if p.get("answer_jaccard") is not None and p["answer_jaccard"] > JAC_MIN]
        if not pos:
            continue
        c = min(CAP, len(pos))
        by_bucket[bucket(c)].append((e, pos))

    print("available / target per bucket:")
    for b in BUCKETS:
        print(f"  {b:>6}: avail={len(by_bucket[b]):>3}  target={TARGET[b]}")

    # allocate high->low, spilling any deficit into the next lower bucket
    chosen, deficit = [], 0
    for b in BUCKETS:
        want = TARGET[b] + deficit
        pool = by_bucket[b]
        take = min(want, len(pool))
        deficit = want - take
        idx = rng.choice(len(pool), size=take, replace=False) if take else []
        for i in idx:
            chosen.append(pool[i])
    print(f"selected {len(chosen)} anchors (target 462; leftover deficit {deficit})")

    out = []
    for e, pos in chosen:
        # cap at 20, keeping the highest-jaccard positives (deterministic)
        pos = sorted(pos, key=lambda p: (-p["answer_jaccard"], p["query"]))[:CAP]
        out.append({
            "anchor_query": e["anchor_query"],
            "positive_queries": [
                {"query": p["query"], "answer_jaccard": round(float(p["answer_jaccard"]), 4)}
                for p in pos],
            "negative_queries": [
                {"query": n["query"],
                 "answer_jaccard": round(float(n.get("answer_jaccard", 0.0)), 4)}
                for n in e["negative_queries"]],
        })

    out.sort(key=lambda r: -len(r["positive_queries"]))
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)

    npos = [len(r["positive_queries"]) for r in out]
    nneg = [len(r["negative_queries"]) for r in out]
    posj = [p["answer_jaccard"] for r in out for p in r["positive_queries"]]
    hist = defaultdict(int)
    for c in npos:
        hist[bucket(c)] += 1
    corpus = set()
    for r in out:
        corpus.add(r["anchor_query"])
        corpus.update(p["query"] for p in r["positive_queries"])
        corpus.update(n["query"] for n in r["negative_queries"])
    print(f"\nwrote {len(out)} anchors to {OUT}")
    print(f"  positives/anchor: min {min(npos)} mean {np.mean(npos):.2f} "
          f"median {int(np.median(npos))} max {max(npos)}")
    print(f"  hist: " + "  ".join(f"{b}:{hist[b]}" for b in ['1','2-4','5-9','10-19','20']))
    print(f"  negatives/anchor: min {min(nneg)} max {max(nneg)}")
    print(f"  positive jaccard: min {min(posj):.3f} (all > {JAC_MIN})")
    print(f"  unique corpus queries: {len(corpus)}")


if __name__ == "__main__":
    main()
