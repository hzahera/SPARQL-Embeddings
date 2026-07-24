#!/usr/bin/env python3
"""Answer-set size distribution for both KGs.

Tests the hypothesis that the bimodal answer_jaccard (values collapse to exactly
0 or exactly 1) is driven by small/singleton result sets: when answer sets are
singletons or identical, Jaccard can only be 0 (disjoint) or 1 (equal); partial
overlap requires multi-element sets.

DBpedia result-sets fit in memory; Wikidata (~6 GB) is streamed with ijson so we
never hold the whole file. We only ever measure len(answer_set).

Run:  python scripts/analyze_answerset_sizes.py
"""
import json
from collections import Counter
from pathlib import Path

import ijson

DS = Path(__file__).resolve().parents[1] / "datasets"


def summarize(sizes: Counter, label: str):
    total = sum(sizes.values())
    n1 = sizes.get(1, 0)
    n0 = sizes.get(0, 0)
    le2 = sum(v for k, v in sizes.items() if k <= 2)
    le5 = sum(v for k, v in sizes.items() if k <= 5)
    # weighted mean / median
    flat_median_pos = total // 2
    running, median = 0, None
    for k in sorted(sizes):
        running += sizes[k]
        if running >= flat_median_pos and median is None:
            median = k
    mean = sum(k * v for k, v in sizes.items()) / total
    print(f"\n{label}  ({total} distinct result sets)")
    print(f"  size == 0 (empty)   : {n0/total:.1%}")
    print(f"  size == 1 (singleton): {n1/total:.1%}")
    print(f"  size <= 2           : {le2/total:.1%}")
    print(f"  size <= 5           : {le5/total:.1%}")
    print(f"  median size         : {median}")
    print(f"  mean size           : {mean:.1f}")
    print(f"  max size            : {max(sizes)}")


def dbpedia():
    d = json.load(open(DS / "result_sets_dbpedia.json"))
    sizes = Counter(len(v) for v in d.values())
    summarize(sizes, "DBPEDIA")


def wikidata():
    sizes = Counter()
    path = DS / "result_sets_wikidata.json"
    with open(path, "rb") as fh:
        # top level is an object: query_text -> [bindings]; count array lengths
        for _key, arr in ijson.kvitems(fh, ""):
            sizes[len(arr)] += 1
    summarize(sizes, "WIKIDATA (streamed)")


if __name__ == "__main__":
    dbpedia()
    wikidata()
