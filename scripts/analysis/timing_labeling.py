#!/usr/bin/env python3
"""
Labeling cost for the paper (Section 3.2): time to compute the answer-overlap (Jaccard) relation
for all pairs of usable queries, from the cached answer sets.

Uses the labeling functions of the dataset build script itself
(datasets/scripts/rebuild_<graph>_clean_splits.py): an inverted index over answer rows counts the
shared rows of every pair that shares at least one row (all other pairs have J = 0), and J is then
computed for these pairs and the positive components are formed. No query is executed.

Run from the repository root:
  python scripts/analysis/timing_labeling.py --dataset dbpedia
Output: printed summary + results/analysis/timing_labeling_<dataset>.json
Memory: all cached answer sets are held in memory (as during dataset construction).
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dbpedia", "wikidata"])
    ap.add_argument("--positive-min", type=float, default=0.8)
    args = ap.parse_args()
    g = args.dataset

    spec = importlib.util.spec_from_file_location("rebuild", f"datasets/scripts/rebuild_{g}_clean_splits.py")
    rb = importlib.util.module_from_spec(spec)
    sys.modules["rebuild"] = rb  # needed for the dataclasses in the module
    spec.loader.exec_module(rb)

    cache = Path(f"datasets/{g}-splits/query_execution_cache.jsonl")
    t0 = time.perf_counter()
    answer_sets, statuses = [], {}
    with cache.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            statuses[obj["status"]] = statuses.get(obj["status"], 0) + 1
            if obj["status"] == "success":
                answer_sets.append(set(obj["answers"]))
    t_load = time.perf_counter() - t0
    n = len(answer_sets)
    print(f"{g}: {n} usable queries ({statuses}); loaded cache in {t_load:.1f}s")

    t0 = time.perf_counter()
    inter = rb.build_shared_answer_intersections(answer_sets)
    t_inter = time.perf_counter() - t0

    t0 = time.perf_counter()
    pos, comps = rb.build_positive_edges_and_components(answer_sets, inter, positive_min=args.positive_min)
    t_pos = time.perf_counter() - t0

    total = t_inter + t_pos
    res = {"dataset": g, "usable_queries": n, "all_pairs": n * (n - 1) // 2,
           "pairs_sharing_rows": len(inter), "positive_edges": len(pos), "components": len(comps),
           "seconds_load_cache": t_load, "seconds_intersections": t_inter,
           "seconds_jaccard_and_components": t_pos, "seconds_labeling_total": total}
    print(json.dumps(res, indent=2))
    print(f"\nLabeling all {n*(n-1)//2:,} pairs took {total:.1f}s ({total/60:.1f} min), excluding cache loading.")

    dest = Path("results/analysis") / f"timing_labeling_{g}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(dest, "w"), indent=2)
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()
