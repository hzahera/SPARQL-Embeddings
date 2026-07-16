"""Workload ceiling analysis for the Wikidata caching simulation.

Replays the seed-42 stream with a *perfect* selector: for each query, the
best answer-set Jaccard available in a K=100 always-insert FIFO cache.
This bounds the quality any embedding model can achieve at a given hit
rate, so each model's (hit rate, quality) cell can be read as a fraction
of the achievable ceiling rather than as an absolute.

Run from the repo root (expects datasets/ alongside). The always-insert
cache is a slight upper-bound relaxation of the simulation's
no-insert-on-hit protocol; present results as a ceiling, not an exact
counterfactual.

Usage: python scripts/caching_ceiling_wikidata.py
"""

import json
import random
from collections import deque
from pathlib import Path

import numpy as np

try:
    import ijson  # streams the 6.4 GB oracle without loading it whole
except ImportError as exc:
    raise SystemExit("pip install ijson") from exc

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
K = 100

# (label, hit_rate, quality) cells from results/caching/caching_wikidata.json
# at K=100 — update here if the grid is regenerated.
OPERATING_POINTS = [
    ("SparqlCE  theta=0.7", 0.844, 0.052),
    ("Qwen3-ft  theta=0.7", 0.640, 0.058),
    ("SparqlCE  theta=0.8", 0.647, 0.077),
    ("Qwen3-ft  theta=0.8", 0.600, 0.053),
    ("SparqlCE  theta=0.9", 0.545, 0.081),
    ("Qwen3-ft  theta=0.9", 0.417, 0.119),
]


def normalize_query(q: str) -> str:
    return " ".join(q.split())


def freeze_json(value):
    if isinstance(value, dict):
        return tuple(sorted((k, freeze_json(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(freeze_json(v) for v in value)
    return value


def jaccard(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def main() -> None:
    triplets = json.load(open(ROOT / "datasets" / "wikidata-triplets-dataset.json"))
    splits = json.load(open(ROOT / "datasets" / "splits_wikidata.json"))
    test = [triplets[i]["anchor_query"] for i in splits["test_idx"]]
    train_keys = {normalize_query(triplets[i]["anchor_query"]) for i in splits["train_idx"]}
    del triplets, splits

    wanted = {normalize_query(q) for q in test}
    print(f"test anchors: {len(test)} | unique keys: {len(wanted)}")

    result_sets = {}
    with open(ROOT / "datasets" / "result_sets_wikidata.json", "rb") as f:
        for raw_query, rows in ijson.kvitems(f, ""):
            key = normalize_query(raw_query)
            if key in wanted and rows:
                result_sets[key] = frozenset(freeze_json(r) for r in rows)
    print(f"oracle coverage of test anchors: {len(result_sets)}")

    stream = [q for q in test if normalize_query(q) in result_sets]
    stream_keys = {normalize_query(q) for q in stream}
    print(f"stream size: {len(stream)} | dropped: {len(test) - len(stream)}")
    print(f"unique in stream: {len(stream_keys)}")
    print(f"train/test key overlap (leakage check, expect 0): {len(train_keys & stream_keys)}")

    random.seed(SEED)
    random.shuffle(stream)

    cache = deque(maxlen=K)
    best = []
    for q in stream:
        R = result_sets[normalize_query(q)]
        best.append(max((jaccard(R, c) for c in cache), default=0.0))
        cache.append(R)

    bj = np.sort(np.array(best))[::-1]
    n = len(bj)
    print(f"\noracle nearest-Jaccard in cache (K={K}, always-insert):")
    print(f"  mean={bj.mean():.3f}  median={np.median(bj):.3f}")
    for t in (0.9, 0.5, 0.1):
        print(f"  fraction of queries with a >={t} Jaccard neighbor: {(np.array(best) >= t).mean():.3f}")

    print("\nceiling and efficiency at matched hit rate:")
    for label, hit_rate, quality in OPERATING_POINTS:
        k = int(round(hit_rate * n))
        ceiling = bj[:k].mean() if k else float("nan")
        print(
            f"  {label}: hit={hit_rate:.3f}  achieved={quality:.3f}  "
            f"ceiling={ceiling:.3f}  efficiency={quality / ceiling:.1%}"
        )


if __name__ == "__main__":
    main()
