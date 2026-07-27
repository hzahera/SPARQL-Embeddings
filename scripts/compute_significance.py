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


def check_same_split(runs, label):
    """All runs in a comparison must score the identical anchor list.

    The paired Wilcoxon aligns per-anchor scores positionally, so a differing
    split or anchor order silently produces a meaningless test rather than an
    error.
    """
    splits = {r.get("split") for r in runs}
    if len(splits) > 1:
        raise SystemExit(f"{label}: runs use different --split values: {splits}")
    anchors = runs[0]["anchors"]
    for r in runs[1:]:
        if r["anchors"] != anchors:
            raise SystemExit(
                f"{label}: anchor lists differ across seeds -- re-export every "
                f"run against the identical --split before comparing.")
    return anchors


def seed_average(runs, label):
    """Collapse N per-seed runs into one run whose per-anchor scores are the
    mean across seeds.

    Comparing seed-averaged per-anchor vectors is both more powerful and less
    arbitrary than picking one 'representative' seed per side: it tests the
    method, not a particular training run, and it stops a lucky or unlucky seed
    from deciding the p-value.
    """
    anchors = check_same_split(runs, label)
    per_anchor = {}
    for metric in METRICS:
        mats = [r["per_anchor"][metric] for r in runs if metric in r.get("per_anchor", {})]
        if len(mats) != len(runs):
            continue
        per_anchor[metric] = np.mean(np.array(mats, dtype=float), axis=0).tolist()
    return {
        "anchors": anchors,
        "per_anchor": per_anchor,
        "aggregate": {m: float(np.mean([r["aggregate"][m] for r in runs]))
                      for m in METRICS if all(m in r["aggregate"] for r in runs)},
        "seed": [r.get("seed") for r in runs],
        "model_id": runs[0].get("model_id"),
        "split": runs[0].get("split"),
    }


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
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--baseline",
                     help="single eval_rq1_export.py output for the strongest baseline")
    g.add_argument("--baseline-glob",
                     help="glob matching PER-SEED baseline exports. Preferred: with "
                          "seeds on both sides the test compares seed-averaged "
                          "per-anchor scores, so neither side is represented by one "
                          "arbitrary training run.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sparqlce_runs = load_runs(args.sparqlce_glob)
    cross_seed = cross_seed_stats(sparqlce_runs)

    if args.baseline_glob:
        baseline_runs = load_runs(args.baseline_glob)
        check_same_split(sparqlce_runs + baseline_runs, "sparqlce+baseline")
        sp_avg = seed_average(sparqlce_runs, "sparqlce")
        bl_avg = seed_average(baseline_runs, "baseline")
        sig = paired_test(sp_avg, bl_avg)
        baseline_cross_seed = cross_seed_stats(baseline_runs)
        comparison = "seed-averaged per-anchor scores (both sides multi-seed)"
        baseline_seeds = [r.get("seed") for r in baseline_runs]
        baseline_model = baseline_runs[0].get("model_id")
        rep_seed = None
    else:
        with open(args.baseline) as f:
            baseline_run = json.load(f)
        check_same_split(sparqlce_runs + [baseline_run], "sparqlce+baseline")
        rep_run = representative_run(sparqlce_runs)
        sig = paired_test(rep_run, baseline_run)
        baseline_cross_seed = None
        comparison = ("representative SparqlCE seed vs single-seed baseline "
                      "-- prefer --baseline-glob with per-seed baseline exports")
        baseline_seeds = [baseline_run.get("seed")]
        baseline_model = baseline_run.get("model_id")
        rep_seed = rep_run.get("seed")

    summary = {
        "n_seeds": len(sparqlce_runs),
        "seeds": [r.get("seed") for r in sparqlce_runs],
        "split": sparqlce_runs[0].get("split"),
        "comparison": comparison,
        "cross_seed_stats": cross_seed,
        "representative_seed": rep_seed,
        "baseline_model": baseline_model,
        "baseline_seeds": baseline_seeds,
        "baseline_cross_seed_stats": baseline_cross_seed,
        "significance_vs_baseline": sig,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"n_seeds={summary['n_seeds']}  seeds={summary['seeds']}  "
          f"split={summary['split']}")
    print(f"comparison: {summary['comparison']}")
    print("\nSparqlCE cross-seed mean +/- std:")
    for metric, s in cross_seed.items():
        print(f"  {metric:10s} {s['mean']:.4f} +/- {s['std']:.4f}  (n={s['n_seeds']})")
    if baseline_cross_seed:
        print(f"\nBaseline ({summary['baseline_model']}) cross-seed mean +/- std:")
        for metric, s in baseline_cross_seed.items():
            print(f"  {metric:10s} {s['mean']:.4f} +/- {s['std']:.4f}  (n={s['n_seeds']})")
    print(f"\nSignificance vs. {summary['baseline_model']} "
          f"(baseline seeds={summary['baseline_seeds']}, "
          f"Wilcoxon signed-rank, one-sided):")
    for metric, s in sig.items():
        p = s.get("p_value")
        flag = "*" if s.get("significant_at_0.05") else " "
        print(f"  {metric:10s} p={p}  mean_diff={s.get('mean_diff'):.4f} {flag}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
