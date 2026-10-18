#!/usr/bin/env python3
"""
Checkpoint-selection check: epoch chosen by validation nDCG@10 (what training does)
versus the epoch that maximizes validation nDCG@5 (the reported cutoff).

Reads results/raw/<dataset>/<model>/tuned/seed_*/training_history.json and reports, per model,
how much validation nDCG@5 / Hits@1 is given up by selecting on nDCG@10.

Run from the repository root:
  python scripts/analysis/checkpoint_selection.py
Output: printed table + results/analysis/checkpoint_selection.json
"""
import json
from pathlib import Path

import numpy as np

RAW = Path("results/raw")
out = {}
for ds in ("dbpedia", "wikidata"):
    print(f"\n{ds}")
    print(f"  {'model':16s} {'same epoch':>10s} {'mean loss nDCG@5':>17s} {'max loss nDCG@5':>16s} {'mean loss Hits@1':>17s}")
    out[ds] = {}
    all_losses = []
    for mdir in sorted(p for p in (RAW / ds).iterdir() if p.is_dir()):
        loss5, lossh, same = [], [], 0
        for hist in sorted(mdir.glob("tuned/seed_*/training_history.json")):
            h = [e for e in json.load(open(hist)) if e.get("validation")]
            if not h:
                continue
            v = lambda e, m: e["validation"]["ranking"].get(m, float("nan"))
            sel = max(h, key=lambda e: v(e, "ndcg@10"))
            best = max(h, key=lambda e: v(e, "ndcg@5"))
            same += sel["epoch"] == best["epoch"]
            loss5.append(v(best, "ndcg@5") - v(sel, "ndcg@5"))
            lossh.append(max(v(e, "hits@1") for e in h) - v(sel, "hits@1"))
        if not loss5:
            continue
        all_losses += loss5
        out[ds][mdir.name] = {"seeds": len(loss5), "same_epoch": same,
                              "ndcg5_loss_mean": float(np.mean(loss5)), "ndcg5_loss_max": float(np.max(loss5)),
                              "hits1_loss_mean": float(np.nanmean(lossh))}
        print(f"  {mdir.name:16s} {same:>5d}/{len(loss5):<4d} {np.mean(loss5):>17.4f} {np.max(loss5):>16.4f} {np.nanmean(lossh):>17.4f}")
    if all_losses:
        out[ds]["_all"] = {"ndcg5_loss_mean": float(np.mean(all_losses)), "ndcg5_loss_max": float(np.max(all_losses))}
        print(f"  {'all runs':16s} {'':10s} {np.mean(all_losses):>17.4f} {np.max(all_losses):>16.4f}")

json.dump(out, open("results/analysis/checkpoint_selection.json", "w"), indent=2)
print("\nSaved results/analysis/checkpoint_selection.json")
