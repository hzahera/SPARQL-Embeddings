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
from scipy.stats import wilcoxon, ttest_rel

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


def input_parity(sparqlce_runs, baseline_runs):
    """Surface the eval-time input transforms both sides actually used.

    eval_rq1_export.py records what it fed each model. Runs exported before that
    was recorded have no field at all -- which is exactly the case that needs
    flagging, because that version encoded text baselines on raw opaque Wikidata
    ids while SparqlCE's graph builder resolved labels internally.
    """
    kgs = {r.get("kg") for r in sparqlce_runs + baseline_runs}
    bl = [r.get("input_transform") for r in baseline_runs]
    info = {
        "kg": sorted(k for k in kgs if k),
        "baseline_input_transform": sorted({str(x) for x in bl}),
        "sparqlce_labels_disabled": sorted(
            {bool(r.get("sparqlce_labels_disabled")) for r in sparqlce_runs}),
        "warnings": [],
    }
    if any(x is None for x in bl):
        info["warnings"].append(
            "baseline runs predate input-transform recording -- re-export them "
            "with the current eval_rq1_export.py. The old version gave text "
            "baselines no Wikidata label expansion and no E5 'query: ' prefix, "
            "both of which handicap the baseline relative to SparqlCE.")
    elif "wikidata" in kgs and not any(
            x and "expand_wikidata_labels" in x for x in bl):
        info["warnings"].append(
            "Wikidata run whose text baseline did NOT get label expansion while "
            "SparqlCE resolves labels internally -- not a like-for-like input.")
    return info


def seed_level_test(sparqlce_runs, baseline_runs):
    """Paired test with the SEED as the unit of analysis.

    The anchor-level Wilcoxon on seed-averaged scores answers "does this method
    rank these 450 anchors better", treating anchors as independent and the
    training run as fixed. It therefore cannot see cross-seed variance at all:
    on the first Wikidata sweep SparqlCE ranged 0.571-0.881 MRR@10 across seeds
    (sigma=0.135, ~50x the baseline's) and the anchor-level p was still 1e-32.
    Report BOTH -- this one answers the question a reviewer actually asks, "does
    the method beat the baseline when you retrain it", and it is the one that
    degrades honestly when training is unstable.

    Requires the same seeds on both sides (the sweep guarantees it); pairs on
    the seed value, not on file order.
    """
    sp_by_seed = {r.get("seed"): r for r in sparqlce_runs}
    bl_by_seed = {r.get("seed"): r for r in baseline_runs}
    seeds = sorted(s for s in sp_by_seed.keys() & bl_by_seed.keys() if s is not None)
    if len(seeds) < 3:
        return {"note": f"only {len(seeds)} paired seeds -- seed-level test skipped "
                        f"(needs >=3 to be meaningful)", "seeds": seeds}

    out = {"unit": "seed", "n_seeds": len(seeds), "seeds": seeds}
    for metric in METRICS:
        a = np.array([sp_by_seed[s]["aggregate"][metric] for s in seeds
                      if metric in sp_by_seed[s]["aggregate"]], dtype=float)
        b = np.array([bl_by_seed[s]["aggregate"][metric] for s in seeds
                      if metric in bl_by_seed[s]["aggregate"]], dtype=float)
        if len(a) != len(seeds) or len(b) != len(seeds):
            continue
        t = ttest_rel(a, b)
        # One-sided (SparqlCE > baseline): halve the two-sided p when the sign
        # agrees, else it is the complement.
        p_one = t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2
        out[metric] = {
            "sparqlce_mean": float(a.mean()), "sparqlce_std": float(a.std(ddof=1)),
            "baseline_mean": float(b.mean()), "baseline_std": float(b.std(ddof=1)),
            "mean_diff": float((a - b).mean()),
            "t_statistic": float(t.statistic),
            "p_value_one_sided": float(p_one),
            "significant_at_0.05": bool(p_one < 0.05),
            # Distribution-free and the strongest claim available at n=5: does
            # the WORST SparqlCE seed still beat the BEST baseline seed?
            "worst_sparqlce_beats_best_baseline": bool(a.min() > b.max()),
            "worst_sparqlce": float(a.min()), "best_baseline": float(b.max()),
        }
    return out


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
        seed_level = seed_level_test(sparqlce_runs, baseline_runs)
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
        seed_level = {"note": "single-seed baseline -- a seed-level test needs "
                              "per-seed baseline runs (--baseline-glob)"}

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
        "significance_seed_level": seed_level,
        "input_parity": input_parity(sparqlce_runs, baseline_runs
                                     if args.baseline_glob else [baseline_run]),
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

    if seed_level.get("note"):
        print(f"\nSeed-level test: {seed_level['note']}")
    else:
        print(f"\nSEED-LEVEL paired t-test (unit = seed, n={seed_level['n_seeds']}, "
              f"one-sided) -- report this ALONGSIDE the anchor-level p above; it "
              f"is the one that reflects training instability:")
        for metric in METRICS:
            s = seed_level.get(metric)
            if not s:
                continue
            flag = "*" if s["significant_at_0.05"] else " "
            dom = "  [worst SparqlCE seed > best baseline seed]" \
                if s["worst_sparqlce_beats_best_baseline"] else ""
            print(f"  {metric:10s} {s['sparqlce_mean']:.4f}+/-{s['sparqlce_std']:.4f} vs "
                  f"{s['baseline_mean']:.4f}+/-{s['baseline_std']:.4f}  "
                  f"p={s['p_value_one_sided']:.4g} {flag}{dom}")

    for w in summary["input_parity"]["warnings"]:
        print(f"\n!! INPUT PARITY: {w}")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
