#!/usr/bin/env python3
"""Run where <graph>-clean-splits/query_execution_cache.jsonl exists (output of the rebuild script).
Gives: execution time, pair-labeling time, truncation check (answer sizes that recur like a cap),
and queries whose answers contain blank nodes.  Usage: python table1_from_cache.py <cache.jsonl>"""
import json, sys, time
from collections import Counter, defaultdict
from itertools import combinations
path = sys.argv[1]
by = {}
for line in open(path):
    if line.strip():
        r = json.loads(line); by[r["query_hash"]] = r          # append-only cache: keep last record
rows = list(by.values()); ok = [r for r in rows if r["status"] == "success"]
sizes = sorted(r["n_answers"] for r in ok); c = Counter(sizes)
el = [r["elapsed_s"] for r in rows if r.get("elapsed_s") is not None]
bnode = [r["query"] for r in ok if any('"bnode"' in a for a in r["answers"])]
t0 = time.time(); inv = defaultdict(list); inter = defaultdict(int)
for i, r in enumerate(ok):
    for a in set(r["answers"]): inv[a].append(i)
for ids in inv.values():
    for i, j in combinations(sorted(set(ids)), 2): inter[(i, j)] += 1
out = {"status_counts": dict(Counter(r["status"] for r in rows)),
       "answer_size_min_median_max": [sizes[0], sizes[len(sizes) // 2], sizes[-1]],
       "possible_caps(size>=1000 seen>=3 times)": {s: n for s, n in c.items() if s >= 1000 and n >= 3},
       "queries_with_bnodes": len(bnode),
       "execution_time_sum_s": round(sum(el), 1), "labeling_time_s": round(time.time() - t0, 1)}
print(json.dumps(out, indent=2))
json.dump({"summary": out, "bnode_queries": bnode}, open(path.replace(".jsonl", "_table1.json"), "w"), indent=2)
