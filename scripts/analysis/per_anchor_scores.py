#!/usr/bin/env python3
"""
Per-anchor retrieval scores for every encoder checkpoint (zero-shot and each fine-tuned seed).

Two rankings per checkpoint, from one encoding pass:
  <tag>.json       protocol B, exactly the paper's evaluation (result_aware.evaluation.rank_record_global:
                   rank all test-split queries, unstored candidates are non-relevant); checked against
                   evaluation_test.json.
  <tag>_full.json  protocol C (Table tab:fullcollection): rank all test queries of <dataset>-test-rq3.json,
                   relevance from the complete RQ3 pair labels (J >= 0.8; intermediate/negative = 0),
                   as in nonneural.py.

Run from the repository root:
  export PYTHONPATH="$PWD/src:$PYTHONPATH"
  python scripts/analysis/per_anchor_scores.py --dataset wikidata
  python scripts/analysis/per_anchor_scores.py --dataset wikidata --models mpnet e5-base-v2   # subset

Output: results/per_anchor/<dataset>/<model>/{zero_shot,seed_42,...}{,_full}.json   (skips runs whose two files exist)
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from result_aware.config import load_model_config
from result_aware.data import load_records
from result_aware.evaluation import encode_split, rank_record_global
from result_aware.metrics import hits_at_k, is_relevant, mrr_at_k, ndcg_at_k
from result_aware.model import HFEmbeddingModel
from result_aware.utils import normalize_query

MODELS = ["mpnet", "e5-base-v2", "bge-base", "gte-base", "qwen3-embedding",
          "gemma", "codebert", "graphcodebert", "qwen2.5-coder", "spbert"]
SEEDS = [42, 43, 44, 45, 46]


def score_run(model, records):
    encoded = encode_split(model, records, batch_size=64, show_progress=False)
    out = {}
    for rec in records:
        ranked = rank_record_global(rec, encoded)
        rels = [float(x["answer_jaccard"]) for x in ranked]
        binary = [is_relevant(x, 0.8, "ge") for x in rels]
        out[normalize_query(rec.anchor_query)] = {
            "hits@1": float(hits_at_k(binary, 1)),
            "mrr@5": float(mrr_at_k(binary, 5)),
            "ndcg@5": float(ndcg_at_k(rels, 5, gain="linear")),
        }
    return out


def score_full(model, rq3, anchors):
    """Protocol C: rank all RQ3 test queries, relevance from complete pair labels."""
    qs = [normalize_query(x["query"]) for x in rq3["queries"]]
    idx = {q: i for i, q in enumerate(qs)}
    emb = model.encode([x["query"] for x in rq3["queries"]], batch_size=64).float()
    sim = (emb @ emb.T).numpy()
    full = {}
    for a, c, j in rq3["positive_pairs"]:
        full[(a, c)] = full[(c, a)] = float(j)
    out = {}
    for anchor in anchors:
        a = idx[anchor]
        cands = np.array([c for c in range(len(qs)) if c != a])
        order = cands[np.argsort(-sim[a, cands], kind="stable")]
        rels = [full.get((a, int(c)), 0.0) for c in order]
        rels = [r if r >= 0.8 else 0.0 for r in rels]
        binary = [r >= 0.8 for r in rels]
        out[anchor] = {
            "hits@1": float(hits_at_k(binary, 1)),
            "mrr@5": float(mrr_at_k(binary, 5)),
            "ndcg@5": float(ndcg_at_k(rels, 5, gain="linear")),
            "mrr": float(mrr_at_k(binary, len(binary))),
            "ndcg@10": float(ndcg_at_k(rels, 10, gain="linear")),
        }
    return out


def check(scores, eval_file, tag):
    if not eval_file.exists():
        print(f"   {tag}: no {eval_file.name} to compare")
        return
    ref = json.load(open(eval_file))["ranking"]
    for m in ("hits@1", "mrr@5", "ndcg@5"):
        mine = sum(v[m] for v in scores.values()) / len(scores)
        flag = "ok" if abs(mine - ref[m]) < 1e-6 else f"MISMATCH (reference {ref[m]:.6f})"
        print(f"   {tag}: {m} = {mine:.6f} {flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dbpedia", "wikidata"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    records = load_records(f"datasets/{args.dataset}-splits/{args.dataset}-test.json")
    rq3 = json.load(open(f"datasets/{args.dataset}-splits/{args.dataset}-test-rq3.json"))
    anchors = [normalize_query(r.anchor_query) for r in records]
    raw = Path("results/raw") / args.dataset
    out_root = Path("results/per_anchor") / args.dataset
    print(f"{args.dataset}: {len(records)} test anchors")

    for key in args.models:
        cfg = load_model_config(key)
        runs = [("zero_shot", None, raw / key / "zero_shot" / "evaluation_test.json")]
        for s in args.seeds:
            d = raw / key / "tuned" / f"seed_{s}"
            runs.append((f"seed_{s}", d / "best_model", d / "evaluation_test.json"))
        for tag, ckpt, eval_file in runs:
            dest = out_root / key / f"{tag}.json"
            dest_full = out_root / key / f"{tag}_full.json"
            if dest.exists() and dest_full.exists():
                print(f"[{key}] {tag}: exists, skipped")
                continue
            if ckpt is not None and not ckpt.exists():
                print(f"[{key}] {tag}: no checkpoint at {ckpt}, skipped")
                continue
            print(f"[{key}] {tag}")
            model = HFEmbeddingModel(cfg, device=args.device, checkpoint_path=ckpt)
            model.eval()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                if not dest.exists():
                    scores = score_run(model, records)
                    check(scores, eval_file, f"[{key}] {tag}")
                    json.dump(scores, open(dest, "w"))
                full = score_full(model, rq3, anchors)
            json.dump(full, open(dest_full, "w"))
            print(f"   [{key}] {tag} full collection: " + "  ".join(
                f"{m} {np.mean([v[m] for v in full.values()]):.4f}" for m in ("hits@1", "mrr", "ndcg@10")))
            del model
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
