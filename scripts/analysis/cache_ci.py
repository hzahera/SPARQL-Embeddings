#!/usr/bin/env python3
"""
RQ3 caching: pooled hit accuracy with 95% bootstrap intervals, plus mean Jaccard of scored hits.

Reads the per-stream counts that cache evaluation already stores:
  results/raw/<dataset>/<model>/zero_shot/cache.json
  results/raw/<dataset>/<model>/tuned/seed_*/cache.json
  -> capacities["128"]["test"]["semantic"]["per_stream"]   (one entry per stream order)

Accuracy = sum(correct_semantic_hits) / sum(semantic_hits), pooled over runs (matches the tables).
Interval: ZS resamples the 5 stream orders; FT resamples training seeds, then stream orders
within each drawn seed (two-level bootstrap), 10,000 times.

Run from the repository root:
  python scripts/analysis/cache_ci.py            # K = 128
  python scripts/analysis/cache_ci.py --k 32
Output: printed table + results/analysis/cache_ci.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

B = 10_000
rng = np.random.default_rng(0)


def streams(path, k):
    d = json.load(open(path))
    d = d.get("cache", d)
    return d["capacities"][str(k)]["test"]["semantic"]["per_stream"]


def counts(runs):
    """array (n_runs, 5): correct hits, all hits, queries, Jaccard sum, Jaccard count"""
    return np.array([[r["correct_semantic_hits"], r["semantic_hits"], r["total_queries"],
                      r.get("known_hit_jaccard_sum", 0.0), r.get("known_hit_jaccard_count", 0)] for r in runs], float)


def pooled(c):
    acc = c[:, 0].sum() / c[:, 1].sum() if c[:, 1].sum() else float("nan")
    chr_ = c[:, 0].sum() / c[:, 2].sum()
    mj = c[:, 3].sum() / c[:, 4].sum() if c[:, 4].sum() else float("nan")
    return acc, chr_, mj


def boot(groups):
    """groups: list of (n_streams, 5) arrays, one per training seed (one group for ZS)."""
    stats = []
    for _ in range(B):
        gs = [groups[i] for i in rng.integers(0, len(groups), len(groups))] if len(groups) > 1 else groups
        c = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in gs])
        stats.append(pooled(c))
    stats = np.array(stats)
    return np.nanpercentile(stats, [2.5, 97.5], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=128)
    args = ap.parse_args()
    raw = Path("results/raw")
    out = {}
    for ds in ("dbpedia", "wikidata"):
        print(f"\n{ds} (K={args.k})")
        print(f"  {'model':16s} {'':3s} {'Acc. [95% CI]':>24s} {'CHR [95% CI]':>24s} {'mean J of hits':>15s}")
        out[ds] = {}
        for mdir in sorted(p for p in (raw / ds).iterdir() if p.is_dir()):
            zs = mdir / "zero_shot" / "cache.json"
            seeds = sorted(mdir.glob("tuned/seed_*/cache.json"))
            res = {}
            for tag, files in (("ZS", [zs] if zs.exists() else []), ("FT", seeds)):
                if not files:
                    continue
                groups = [counts(streams(f, args.k)) for f in files]
                acc, chr_, mj = pooled(np.concatenate(groups))
                lo, hi = boot(groups)
                res[tag] = {"acc": acc, "acc_ci": [lo[0], hi[0]], "chr": chr_, "chr_ci": [lo[1], hi[1]],
                            "mean_jaccard_hits": mj, "runs": int(sum(len(g) for g in groups))}
                print(f"  {mdir.name:16s} {tag:3s} {acc:.3f} [{lo[0]:.3f}, {hi[0]:.3f}]  "
                      f"{chr_:.3f} [{lo[1]:.3f}, {hi[1]:.3f}]  {mj:15.3f}")
            out[ds][mdir.name] = res
    json.dump(out, open("results/analysis/cache_ci.json", "w"), indent=2, default=float)
    print("\nSaved results/analysis/cache_ci.json")


if __name__ == "__main__":
    main()
