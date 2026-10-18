#!/usr/bin/env python3
"""
Paired bootstrap confidence intervals and seen/unseen-template analysis from per-anchor scores.

Inputs (run from the repository root):
  results/per_anchor/<dataset>/<model>/{zero_shot,seed_*}.json     (per_anchor_scores.py)
  results/per_anchor/<dataset>/baselines.json                      (nonneural.py, protocol B)
  scripts/audit/audit_<dataset>_perquery.json                      (audit_splits.py; for templates)
  datasets/<dataset>-splits/<dataset>-{train,test}.json

Fine-tuned per-anchor scores are averaged over the available seeds before bootstrapping, so
intervals reflect variation over test anchors (resampled 10,000 times, paired).

Usage: python scripts/analysis/paired_analysis.py --dataset wikidata
Output: printed tables + results/analysis/paired_analysis_<dataset>.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

METRICS = ["hits@1", "mrr@5", "ndcg@5"]
B = 10_000
ws = lambda q: re.sub(r"\s+", " ", q.strip())


def boot_ci(diff, rng):
    """Mean and 95% percentile interval of a per-anchor difference vector (paired bootstrap)."""
    n = len(diff)
    idx = rng.integers(0, n, size=(B, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dbpedia", "wikidata"])
    ap.add_argument("--baseline", default="tfidf", help="baseline to compare against (tfidf, bm25, ...)")
    args = ap.parse_args()
    g = args.dataset
    root = Path("results/per_anchor") / g
    rng = np.random.default_rng(0)

    test = json.load(open(f"datasets/{g}-splits/{g}-test.json"))
    anchors = [ws(r["anchor_query"]) for r in test]

    def vec(scores, m):
        return np.array([scores[a][m] for a in anchors], dtype=float)

    # ---- encoders
    enc = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        zs = d / "zero_shot.json"
        seeds = sorted(p for p in d.glob("seed_*.json") if re.fullmatch(r"seed_\d+\.json", p.name))
        if not zs.exists() or not seeds:
            print(f"skip {d.name}: zero_shot or seeds missing")
            continue
        z = json.load(open(zs))
        ft = [json.load(open(s)) for s in seeds]
        enc[d.name] = {m: (vec(z, m), np.mean([vec(f, m) for f in ft], axis=0)) for m in METRICS}
        enc[d.name]["n_seeds"] = len(seeds)

    # ---- baseline
    base = None
    bfile = root / "baselines.json"
    if bfile.exists():
        bl = json.load(open(bfile))
        key = f"B:{args.baseline}"
        if key in bl:
            base = {m: np.array([bl[key][a][m] for a in anchors], dtype=float) for m in METRICS}
    if base is None:
        print(f"no per-anchor baseline '{args.baseline}' found in {bfile}; baseline comparisons skipped")

    # ---- templates (seen / unseen)
    groups = None
    pq = Path(f"scripts/audit/audit_{g}_perquery.json")
    if pq.exists():
        info = json.load(open(pq))
        train = json.load(open(f"datasets/{g}-splits/{g}-train.json"))
        train_t = {info[ws(c)]["template"] for r in train
                   for c in [r["anchor_query"]] + [x["query"] for x in r["positive_queries"] + r["negative_queries"]]
                   if ws(c) in info}
        groups = np.array([info[a]["template"] in train_t for a in anchors])
        print(f"templates: {int(groups.sum())} seen, {int((~groups).sum())} unseen test anchors")

    out = {"dataset": g, "anchors": len(anchors), "bootstrap_resamples": B, "baseline": args.baseline}

    # ---- 1. fine-tuning gains (FT - ZS)
    print(f"\n[1] Fine-tuning gain (FT mean over seeds - ZS), paired 95% CI over {len(anchors)} anchors")
    out["gain"] = {}
    for name, e in enc.items():
        row = {m: boot_ci(e[m][1] - e[m][0], rng) for m in METRICS}
        out["gain"][name] = row
        print(f"  {name:16s} " + "  ".join(f"{m} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for m, v in row.items()))
    allz = {m: np.mean([e[m][0] for e in enc.values()], axis=0) for m in METRICS}
    allf = {m: np.mean([e[m][1] for e in enc.values()], axis=0) for m in METRICS}
    out["gain"]["_mean_over_encoders"] = {m: boot_ci(allf[m] - allz[m], rng) for m in METRICS}
    print("  " + f"{'mean of encoders':16s} " + "  ".join(
        f"{m} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for m, v in out["gain"]["_mean_over_encoders"].items()))

    # ---- 2. fine-tuned vs baseline
    if base is not None:
        print(f"\n[2] Fine-tuned minus {args.baseline}, paired 95% CI")
        out["vs_baseline"] = {}
        for name, e in enc.items():
            row = {m: boot_ci(e[m][1] - base[m], rng) for m in METRICS}
            out["vs_baseline"][name] = row
            sig = "above" if row["hits@1"][1] > 0 else ("below" if row["hits@1"][2] < 0 else "n.s.")
            print(f"  {name:16s} " + "  ".join(f"{m} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for m, v in row.items())
                  + f"   Hits@1: {sig}")
        print("\n    zero-shot minus baseline (Hits@1)")
        out["zs_vs_baseline"] = {}
        for name, e in enc.items():
            v = boot_ci(e["hits@1"][0] - base["hits@1"], rng)
            out["zs_vs_baseline"][name] = v
            print(f"  {name:16s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")

    # ---- 3. seen vs unseen templates
    if groups is not None:
        print("\n[3] Seen vs. unseen templates (mean Hits@1 / nDCG@5; gain with 95% CI)")
        out["templates"] = {"seen_anchors": int(groups.sum()), "unseen_anchors": int((~groups).sum()), "by_model": {}}
        for label, mask in (("seen", groups), ("unseen", ~groups)):
            print(f"  -- {label} ({int(mask.sum())} anchors)")
            if base is not None:
                out["templates"].setdefault("baseline", {})[label] = {
                    m: float(base[m][mask].mean()) for m in ("hits@1", "ndcg@5")}
                print(f"     {args.baseline:16s} Hits@1 {base['hits@1'][mask].mean():.3f}  nDCG@5 {base['ndcg@5'][mask].mean():.3f}")
            for name, e in list(enc.items()) + [("mean of encoders", {m: (allz[m], allf[m]) for m in METRICS})]:
                r = {}
                for m in ("hits@1", "ndcg@5"):
                    z, f = e[m][0][mask], e[m][1][mask]
                    r[m] = {"zs": float(z.mean()), "ft": float(f.mean()), "gain_ci": boot_ci(f - z, rng)}
                out["templates"]["by_model"].setdefault(name, {})[label] = r
                h = r["hits@1"]
                print(f"     {name:16s} Hits@1 {h['zs']:.3f} -> {h['ft']:.3f}  gain {h['gain_ci'][0]:+.3f} "
                      f"[{h['gain_ci'][1]:+.3f}, {h['gain_ci'][2]:+.3f}]   nDCG@5 {r['ndcg@5']['zs']:.3f} -> {r['ndcg@5']['ft']:.3f}")

    # ---- 4. full-collection table (protocol C, tab:fullcollection)
    FULL = ["hits@1", "mrr", "ndcg@10"]
    rows = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        zs = d / "zero_shot_full.json"
        seeds = sorted(d.glob("seed_*_full.json"))
        if not zs.exists() or not seeds:
            continue
        z = json.load(open(zs))
        ft = [json.load(open(s)) for s in seeds]
        rows[d.name] = {m: (float(np.mean(vec(z, m))), float(np.mean([vec(f, m).mean() for f in ft])),
                            float(np.std([vec(f, m).mean() for f in ft], ddof=1)) if len(ft) > 1 else 0.0)
                        for m in FULL}
    if rows:
        print("\n[4] Full collection (protocol C): ZS -> FT (seed std)")
        for name, r in rows.items():
            print(f"  {name:16s} " + "  ".join(f"{m} {v[0]:.3f} -> {v[1]:.3f} ({v[2]:.3f})" for m, v in r.items()))
        mean = {m: (float(np.mean([r[m][0] for r in rows.values()])), float(np.mean([r[m][1] for r in rows.values()])))
                for m in FULL}
        print(f"  {'mean of encoders':16s} " + "  ".join(f"{m} {v[0]:.3f} -> {v[1]:.3f}" for m, v in mean.items()))
        out["full_collection"] = {"by_model": rows, "mean_over_encoders": mean}
        bl = json.load(open(bfile)) if bfile.exists() else {}
        ckey = f"C:{args.baseline}"
        if ckey in bl:
            bC = {m: np.array([bl[ckey][a][m] for a in anchors], dtype=float) for m in FULL}
            print(f"\n    full collection: fine-tuned minus {args.baseline} (paired 95% CI); {args.baseline} "
                  + "  ".join(f"{m} {bC[m].mean():.3f}" for m in FULL))
            out["full_collection"]["baseline"] = {m: float(bC[m].mean()) for m in FULL}
            out["full_collection"]["vs_baseline"] = {}
            for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name in rows):
                ft = np.mean([vec(json.load(open(s)), "hits@1") for s in sorted(d.glob("seed_*_full.json"))], axis=0)
                v = boot_ci(ft - bC["hits@1"], rng)
                out["full_collection"]["vs_baseline"][d.name] = v
                print(f"  {d.name:16s} Hits@1 {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")
        else:
            print(f"    (no per-anchor '{ckey}' in {bfile}; full-collection baseline comparison skipped)")

    dest = Path("results/analysis") / f"paired_analysis_{g}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()