#!/usr/bin/env python3
"""Seen/unseen-template analysis on Wikidata (paper, Section 4.3), with the current
template groups in datasets/statistics/wikidata/test_anchor_template_groups_so.json
(392 seen, 450 unseen test anchors).

For every encoder: zero-shot checkpoint and the five fine-tuned seeds in results/.
Ranks all test queries per anchor; Hits@1 with complete labels (J >= 0.8 from
*-test-rq3.json) and with stored labels. Reports per group:
  mean FT gain (FT - ZS) with paired 95% bootstrap CI over anchors (encoders averaged),
  encoders whose gain CI excludes zero, mean FT Hits@1, and TF-IDF for reference.

Usage (repo root, GPU machine with results/):
  python scripts/eval_seen_unseen.py
Output: scripts/seen_unseen_results.json (+ printed summary for the paper)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "syntax_control"))
from eval_syntax_control import bootstrap, load_graph, per_anchor_scores, ws  # noqa: E402
from result_aware.config import load_model_config, load_registry  # noqa: E402
from result_aware.model import HFEmbeddingModel  # noqa: E402


def encode(model_key, ckpt, queries, device):
    cfg = load_model_config(model_key, ROOT / "configs")
    m = HFEmbeddingModel(cfg, device=device, checkpoint_path=ckpt)
    e = m.encode(queries, batch_size=64, show_progress=False)
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", nargs="+", default=None, help="default: all in registry")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()

    g = "wikidata"
    queries, full, anchors = load_graph(ROOT, g)
    test = json.load(open(ROOT / f"datasets/{g}-splits/{g}-test.json", encoding="utf-8"))
    groups_raw = json.load(open(ROOT / f"datasets/statistics/{g}/test_anchor_template_groups_so.json"))
    groups = {ws(k): v for k, v in groups_raw.items()}
    grp = np.array([groups[ws(r["anchor_query"])] for r in test])
    models = args.models or list(load_registry(ROOT / "configs")["models"])
    cache = ROOT / "scripts" / "seen_unseen_cache"
    cache.mkdir(exist_ok=True)

    def h1(model_key, tag, ckpt):
        f = cache / f"{model_key}__{tag}.json"
        if f.exists():
            return json.load(open(f))
        s = per_anchor_scores(encode(model_key, ckpt, queries, args.device), full, anchors)
        out = {"complete": s["complete"]["hits@1"], "stored": s["stored"]["hits@1"]}
        json.dump(out, open(f, "w"))
        return out

    per = {}
    for mk in models:
        zs = h1(mk, "zs", None)
        ft = [h1(mk, f"seed{s}", ROOT / args.results / "raw" / g / mk / "tuned" / f"seed_{s}" / "best_model")
              for s in args.seeds]
        per[mk] = {p: (np.array(zs[p]), np.mean([np.array(x[p]) for x in ft], axis=0)) for p in ("complete", "stored")}
        print("done", mk, flush=True)

    report = {"groups": {k: int((grp == k).sum()) for k in ("seen", "unseen")}}
    for p in ("complete", "stored"):
        report[p] = {}
        for k in ("seen", "unseen"):
            m = grp == k
            ZS = np.mean([per[mk][p][0][m] for mk in models], axis=0)
            FT = np.mean([per[mk][p][1][m] for mk in models], axis=0)
            lo, hi = bootstrap(FT - ZS, args.bootstrap)
            enc = {}
            n_sig = 0
            for mk in models:
                d = per[mk][p][1][m] - per[mk][p][0][m]
                l2, h2 = bootstrap(d, args.bootstrap)
                n_sig += int(l2 > 0)
                enc[mk] = {"gain": round(float(d.mean()), 4), "ci95": [round(l2, 4), round(h2, 4)]}
            report[p][k] = {"zs": round(float(ZS.mean()), 4), "ft": round(float(FT.mean()), 4),
                            "gain": round(float((FT - ZS).mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                            "encoders_ci_above_zero": n_sig, "per_encoder": enc}
    out = ROOT / "scripts" / "seen_unseen_results.json"
    json.dump(report, open(out, "w"), indent=2)
    for p in ("complete", "stored"):
        print(f"\n[{p} labels]")
        for k in ("seen", "unseen"):
            r = report[p][k]
            print(f"{k:7s} n={report['groups'][k]}  gain {r['gain']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]"
                  f"  FT {r['ft']:.2f}  encoders with CI>0: {r['encoders_ci_above_zero']}/{len(models)}")
    print("TF-IDF Hits@1 (same groups, computed separately): seen 0.16, unseen 0.55")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
