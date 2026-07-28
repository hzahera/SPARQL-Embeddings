#!/usr/bin/env python3
"""Summarize training runs from their run_metadata.json files.

Answers the questions that decide whether a multi-seed result is trustworthy:
did every seed get the same budget, did any seed stop while still improving,
and did early stopping fire at a noisy plateau rather than at convergence?

    python scripts/inspect_runs.py --kg wikidata
    python scripts/inspect_runs.py --glob "checkpoints/**/run_metadata.json"

A seed whose best_epoch is at (or within one of) the epoch cap was still
improving when training ended -- the cap, not convergence, set its result.
A seed whose run ended exactly patience epochs after its best stopped on the
early-stopping rule; if the val curve is noisy relative to --min-delta, that
may be a premature stop rather than a plateau.
"""
import argparse
import glob as globmod
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(pattern):
    out = []
    for p in sorted(globmod.glob(pattern, recursive=True)):
        try:
            m = json.load(open(p))
        except (OSError, ValueError):
            continue
        hist = m.get("history") or []
        n_ep = len(hist) or m.get("epochs_ran") or m.get("epochs_run") or None
        out.append({
            "path": p,
            "kind": "e5ft" if ("e5_ft" in p or m.get("model")) else "sparqlce",
            "kg": m.get("kg"), "seed": m.get("seed"),
            "best_epoch": m.get("best_epoch"),
            "epochs_ran": n_ep,
            "epochs_cap": m.get("epochs_max") or m.get("epochs"),
            "patience": m.get("patience"),
            "batch": m.get("batch_size") or m.get("effective_batch"),
            "lr": m.get("lr") or m.get("lr_text"),
            "best_val": (m.get("best_val_mrr")
                         or (m.get("val_metrics") or {}).get("mrr")
                         or (m.get("val_metrics") or {}).get("val_mrr")),
            "history": hist,
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg", default=None)
    ap.add_argument("--glob", default=None, help="override the metadata glob")
    args = ap.parse_args()

    pat = args.glob or str(ROOT / "checkpoints" / "**" / "run_metadata.json")
    rs = rows(pat)
    if args.kg:
        rs = [r for r in rs if r["kg"] == args.kg]
    if not rs:
        raise SystemExit(f"no run_metadata.json matched {pat}")

    hdr = (f"{'kind':9s} {'seed':>4} {'best_ep':>7} {'ran':>4} {'cap':>4} "
           f"{'pat':>4} {'batch':>6} {'lr':>8} {'best_val':>9}  flag")
    print(hdr)
    print("-" * len(hdr))
    for r in rs:
        flag = ""
        be, ran, cap, pat_ = r["best_epoch"], r["epochs_ran"], r["epochs_cap"], r["patience"]
        if be and cap and be >= cap - 1:
            flag = "HIT CAP - still improving"
        elif be and ran and pat_ and ran - be <= pat_:
            flag = f"early-stopped {ran - be} ep after best"
        lr = f"{r['lr']:.1e}" if isinstance(r["lr"], (int, float)) else "-"
        bv = f"{r['best_val']:.4f}" if isinstance(r["best_val"], (int, float)) else "-"
        print(f"{r['kind']:9s} {str(r['seed']):>4} {str(be):>7} {str(ran):>4} "
              f"{str(cap):>4} {str(pat_):>4} {str(r['batch']):>6} {lr:>8} {bv:>9}  {flag}")

    for kind in ("sparqlce", "e5ft"):
        grp = [r for r in rs if r["kind"] == kind and isinstance(r["best_val"], (int, float))]
        if len(grp) < 2:
            continue
        v = [r["best_val"] for r in grp]
        be = [r["best_epoch"] for r in grp if r["best_epoch"]]
        print(f"\n{kind}: best_val {st.mean(v):.4f} +- {st.stdev(v):.4f}  "
              f"(min {min(v):.4f} max {max(v):.4f})")
        if be:
            print(f"{' ' * len(kind)}  best_epoch min {min(be)} max {max(be)} "
                  f"median {int(st.median(be))}")
        caps = {r["epochs_cap"] for r in grp}
        bats = {r["batch"] for r in grp}
        lrs = {r["lr"] for r in grp}
        if len(caps) > 1 or len(bats) > 1 or len(lrs) > 1:
            print(f"{' ' * len(kind)}  WARNING: seeds differ in budget/config -- "
                  f"caps={caps} batches={bats} lrs={lrs}")


if __name__ == "__main__":
    main()
