#!/usr/bin/env python3
"""
RQ3 threshold calibration (Appendix B, validation-optimal procedure).

For every encoder and run (zero-shot and each fine-tuned seed):
  1. replay the VALIDATION stream (FIFO cache, K=128, five stream orders) for every theta in the grid,
     using the complete validation pair labels (<graph>-valid-rq3.json);
  2. select theta* = the theta with the highest validation CHR among those with pooled validation
     hit accuracy >= target (default 0.9); if no theta reaches the target, the run is marked
     infeasible and the theta with the highest validation accuracy is used;
  3. replay the TEST stream at every theta (risk-coverage curve) and report test accuracy / CHR at theta*.
The replay is the unchanged code of result_aware.cache_eval. As a self-check, the test replay at
theta = 0.8 is compared with the stored cache.json of the run (pooled hit counts must be identical).

Run from the repository root (DBpedia checkpoints were saved with transformers 5.x -> use ~/tf5):
  export PYTHONPATH="$PWD/src:$PYTHONPATH"
  ~/tf5/bin/python scripts/analysis/calibrate_thresholds.py --dataset dbpedia
  python          scripts/analysis/calibrate_thresholds.py --dataset wikidata
Output: printed summary + results/analysis/calibration_<dataset>.json (resumable per run)
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from result_aware import cache_eval as ce
from result_aware.config import load_model_config
from result_aware.data import load_records
from result_aware.model import HFEmbeddingModel

MODELS = ["mpnet", "e5-base-v2", "bge-base", "gte-base", "qwen3-embedding",
          "gemma", "codebert", "graphcodebert", "qwen2.5-coder", "spbert"]
SEEDS = [42, 43, 44, 45, 46]
GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99]
STREAM_SEEDS = [100, 101, 102, 103, 104]


def pooled(runs):
    c = sum(r["correct_semantic_hits"] for r in runs)
    h = sum(r["semantic_hits"] for r in runs)
    q = sum(r["total_queries"] for r in runs)
    f = sum(r["false_semantic_hits"] for r in runs)
    return {"correct": c, "hits": h, "queries": q, "false": f,
            "acc": c / h if h else None, "chr": c / q, "shr": h / q, "fhr": f / q}


def curve(ctx, capacity):
    orders = [ce._make_stream_order(len(ctx.queries), s, True) for s in STREAM_SEEDS]
    return {f"{t:.2f}": pooled([ce._simulate_semantic_stream(context=ctx, order=o, capacity=capacity, threshold=t)
                                for o in orders]) for t in GRID}


def select(valid_curve, target):
    feas = [(v["chr"], v["acc"], t) for t, v in valid_curve.items() if v["acc"] is not None and v["acc"] >= target]
    if feas:
        return max(feas)[2], True
    accs = [(v["acc"] if v["acc"] is not None else -1, float(t), t) for t, v in valid_curve.items()]
    return max(accs)[2], False


def stored_counts(cache_json, capacity):
    d = json.load(open(cache_json))
    d = d.get("cache", d)
    return pooled(d["capacities"][str(capacity)]["test"]["semantic"]["per_stream"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dbpedia", "wikidata"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--capacity", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    g = args.dataset
    sd = Path(f"datasets/{g}-splits")
    valid = load_records(str(sd / f"{g}-valid.json"))
    test = load_records(str(sd / f"{g}-test.json"))
    cache_cfg = {"rq3_test_files": [str(sd / f"{g}-valid-rq3.json"), str(sd / f"{g}-test-rq3.json")]}

    dest = Path("results/analysis") / f"calibration_{g}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = json.load(open(dest)) if dest.exists() else {"dataset": g, "target_accuracy": args.target,
                                                        "capacity": args.capacity, "grid": GRID, "runs": {}}
    raw = Path("results/raw") / g

    for key in args.models:
        cfg = load_model_config(key)
        runs = [("zero_shot", None, raw / key / "zero_shot")] + \
               [(f"seed_{s}", raw / key / "tuned" / f"seed_{s}" / "best_model", raw / key / "tuned" / f"seed_{s}")
                for s in args.seeds]
        for tag, ckpt, rdir in runs:
            rid = f"{key}/{tag}"
            if rid in out["runs"]:
                print(f"[{rid}] done, skipped")
                continue
            if ckpt is not None and not ckpt.exists():
                print(f"[{rid}] no checkpoint, skipped")
                continue
            model = HFEmbeddingModel(cfg, device=args.device, checkpoint_path=ckpt)
            model.eval()
            with torch.no_grad():
                vctx = ce._build_online_context(model=model, records=valid, batch_size=64, cache_cfg=cache_cfg,
                                                positive_jaccard_min=0.8, negative_jaccard_max=0.1)
                tctx = ce._build_online_context(model=model, records=test, batch_size=64, cache_cfg=cache_cfg,
                                                positive_jaccard_min=0.8, negative_jaccard_max=0.1)
            vc, tc = curve(vctx, args.capacity), curve(tctx, args.capacity)
            theta, feasible = select(vc, args.target)

            check = "no cache.json"
            cj = rdir / "cache.json"
            if cj.exists():
                ref = stored_counts(cj, args.capacity)
                mine = tc["0.80"]
                check = "ok" if (ref["correct"], ref["hits"]) == (mine["correct"], mine["hits"]) else \
                    f"MISMATCH (stored correct/hits {ref['correct']}/{ref['hits']}, replay {mine['correct']}/{mine['hits']})"
            out["runs"][rid] = {"model": key, "run": tag, "theta": float(theta), "feasible": feasible,
                                "valid_at_theta": vc[theta], "test_at_theta": tc[theta],
                                "valid_curve": vc, "test_curve": tc, "check_theta_0.80": check}
            t = tc[theta]
            acc = f"{t['acc']:.3f}" if t["acc"] is not None else "  -  "
            print(f"[{rid}] theta*={theta}{'' if feasible else ' (target not reached)'}  "
                  f"valid acc {vc[theta]['acc'] if vc[theta]['acc'] is None else round(vc[theta]['acc'], 3)}  "
                  f"test acc {acc} CHR {t['chr']:.3f}  | theta=0.80 replay vs stored: {check}", flush=True)
            json.dump(out, open(dest, "w"), indent=1)
            del model, vctx, tctx
            gc.collect()
            torch.cuda.empty_cache()

    # summary: ZS per model; FT pooled over seeds
    print(f"\n{g}: target validation accuracy {args.target}, K={args.capacity}")
    print(f"{'model':16s} {'ZS theta':>8s} {'ZS acc':>7s} {'ZS CHR':>7s}   {'FT thetas':22s} {'FT acc':>7s} {'FT CHR':>7s} {'met':>4s}")
    summary = {}
    for key in args.models:
        z = out["runs"].get(f"{key}/zero_shot")
        ft = [v for k, v in out["runs"].items() if k.startswith(key + "/seed_")]
        if not ft:
            continue
        c = sum(v["test_at_theta"]["correct"] for v in ft)
        h = sum(v["test_at_theta"]["hits"] for v in ft)
        q = sum(v["test_at_theta"]["queries"] for v in ft)
        s = {"ft_thetas": [v["theta"] for v in ft], "ft_feasible": sum(v["feasible"] for v in ft),
             "ft_test_acc": c / h if h else None, "ft_test_chr": c / q,
             "ft_test_meets_target": [bool(v["test_at_theta"]["acc"] is not None and v["test_at_theta"]["acc"] >= args.target) for v in ft]}
        if z:
            s.update({"zs_theta": z["theta"], "zs_feasible": z["feasible"],
                      "zs_test_acc": z["test_at_theta"]["acc"], "zs_test_chr": z["test_at_theta"]["chr"]})
        summary[key] = s
        fa = f"{s['ft_test_acc']:.3f}" if s["ft_test_acc"] is not None else "  -  "
        za = f"{s['zs_test_acc']:.3f}" if z and s["zs_test_acc"] is not None else "  -  "
        print(f"{key:16s} {s.get('zs_theta', float('nan')):8.2f} {za:>7s} {s.get('zs_test_chr', float('nan')):7.3f}   "
              f"{','.join(f'{t:.2f}' for t in s['ft_thetas']):22s} {fa:>7s} {s['ft_test_chr']:7.3f} "
              f"{sum(s['ft_test_meets_target'])}/{len(ft)}")
    out["summary"] = summary
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()
