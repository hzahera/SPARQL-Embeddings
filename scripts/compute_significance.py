#!/usr/bin/env python3
"""Aggregate multi-seed results and test significance vs. the best baseline.

Two things a single-point-estimate table can't show, which this produces:
  1. Cross-seed variance for SparqlCE itself: mean +/- std of MRR/nDCG over
     N seeds, per KG. Answers "is this one lucky seed?" directly.
  2. Paired significance vs. the strongest baseline: a Wilcoxon signed-rank
     test on per-anchor MRR/nDCG (SparqlCE vs. baseline on the *same* anchors),
     using the seed whose val MRR is closest to the cross-seed mean (a
     representative seed rather than cherry-picking the best one). Reports
     the p-value alongside the effect size.

Usage:
    python scripts/compute_significance.py \
        --sparqlce-glob "results/significance/dbpedia_seed*_full.json" \
        --baseline results/significance/dbpedia_baseline_qwen3-finetuned.json \
        --out results/significance/dbpedia_full_summary.json
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

TOPKS = [3, 5, 10, 20]
METRICS = [f"{m}@{k}" for m in ("mrr", "ndcg") for k in TOPKS]


def load_runs(pattern):
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            runs.append(json.load(f))
    if not runs:
        raise SystemExit(f"no files matched {pattern}")
    return runs


def cross_seed_stats(runs):
    stats = {}
    for metric in METRICS:
        vals = [r["aggregate"][metric] for r in runs if metric in r["aggregate"]]
        if not vals:
            continue
        stats[metric] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
            "values": vals,
        }
    return stats


def representative_run(runs, metric="mrr@10"):
    """Pick the seed whose aggregate score is closest to the cross-seed mean,
    rather than the best seed, so the significance test isn't cherry-picked."""
    vals = [r["aggregate"][metric] for r in runs]
    mean = np.mean(vals)
    idx = int(np.argmin([abs(v - mean) for v in vals]))
    return runs[idx]


def paired_test(sparqlce_run, baseline_run):
    results = {}
    sp_anchors = sparqlce_run["anchors"]
    bl_anchors = baseline_run["anchors"]
    assert sp_anchors == bl_anchors, (
        "anchor lists differ between sparqlce and baseline runs -- "
        "re-export both against the identical eval set before comparing"
    )
    for metric in METRICS:
        if metric not in sparqlce_run["per_anchor"] or metric not in baseline_run["per_anchor"]:
            continue
        a = np.array(sparqlce_run["per_anchor"][metric])
        b = np.array(baseline_run["per_anchor"][metric])
        diff = a - b
        if np.allclose(diff, 0):
            results[metric] = {"p_value": 1.0, "mean_diff": 0.0, "note": "identical scores"}
            continue
        try:
            stat, p = wilcoxon(a, b, alternative="greater", zero_method="wilcox")
        except ValueError as e:
            results[metric] = {"p_value": None, "mean_diff": float(diff.mean()), "note": str(e)}
            continue
        results[metric] = {
            "p_value": float(p),
            "mean_diff": float(diff.mean()),
            "n_anchors": len(a),
            "significant_at_0.05": bool(p < 0.05),
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparqlce-glob", required=True,
                     help="glob matching per-seed eval_rq1_export.py outputs for SparqlCE")
    ap.add_argument("--baseline", required=True,
                     help="single eval_rq1_export.py output for the strongest baseline")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sparqlce_runs = load_runs(args.sparqlce_glob)
    with open(args.baseline) as f:
        baseline_run = json.load(f)

    cross_seed = cross_seed_stats(sparqlce_runs)
    rep_run = representative_run(sparqlce_runs)
    sig = paired_test(rep_run, baseline_run)

    summary = {
        "n_seeds": len(sparqlce_runs),
        "seeds": [r.get("seed") for r in sparqlce_runs],
        "cross_seed_stats": cross_seed,
        "representative_seed": rep_run.get("seed"),
        "baseline_model": baseline_run.get("model_id"),
        "significance_vs_baseline": sig,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"n_seeds={summary['n_seeds']}  seeds={summary['seeds']}")
    print("\nCross-seed mean +/- std:")
    for metric, s in cross_seed.items():
        print(f"  {metric:10s} {s['mean']:.4f} +/- {s['std']:.4f}  (n={s['n_seeds']})")
    print(f"\nSignificance vs. {summary['baseline_model']} (seed={rep_run.get('seed')}, Wilcoxon signed-rank, one-sided):")
    for metric, s in sig.items():
        p = s.get("p_value")
        flag = "*" if s.get("significant_at_0.05") else " "
        print(f"  {metric:10s} p={p}  mean_diff={s.get('mean_diff'):.4f} {flag}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
