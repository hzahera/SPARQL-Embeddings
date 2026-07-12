#!/usr/bin/env python3
"""RQ1 evaluation with PER-ANCHOR score export, for significance testing.

evaluate_rq1.py (notebooks/evaluation-RQ1/) prints only aggregate metrics.
Paired significance testing (see scripts/compute_significance.py) needs the
per-anchor score underneath each aggregate, so this script reuses evaluate_rq1's
data-loading and ranking functions and adds per-anchor MRR/nDCG export instead
of just the mean.

Usage:
    python scripts/eval_rq1_export.py --kg dbpedia \
        --checkpoint checkpoints/dbpedia/seed42_full/best_model.pt \
        --out results/significance/dbpedia_seed42_full.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT_DIR / "notebooks" / "evaluation-RQ1"
for p in (ROOT_DIR, EVAL_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluate_rq1 import (  # noqa: E402
    load_triplets, build_global_corpus, build_eval_items,
    build_binary_relevance_maps, build_emb_map_sparqlce,
    build_emb_map_sentence_transformer, build_sorted_pools_global, dcg,
)
from sparqlce.embedder import SPARQLEmbedder  # noqa: E402

KG_TESTSETS = {
    "dbpedia": ROOT_DIR / "datasets" / "dbpedia_testset_RQ1.json",
    "wikidata": ROOT_DIR / "datasets" / "wikidata_testset_RQ1.json",
}

TOPKS = [3, 5, 10, 20]


def per_anchor_mrr(sorted_pools, items, k):
    out = []
    for sorted_pool, (_, _, positives) in zip(sorted_pools, items):
        if not positives:
            continue
        rr = 0.0
        for rank, q in enumerate(sorted_pool[:k], start=1):
            if q in positives:
                rr = 1.0 / rank
                break
        out.append(rr)
    return out


def per_anchor_ndcg(sorted_pools, rel_maps, k):
    out = []
    for sorted_pool, rel_map in zip(sorted_pools, rel_maps):
        topk = [q for q in sorted_pool[:k] if q in rel_map]
        if not topk:
            out.append(0.0)
            continue
        rels = [rel_map[q] for q in topk]
        dcg_val = dcg(rels)
        ideal_rels = sorted(rel_map.values(), reverse=True)[:k]
        idcg = dcg(ideal_rels)
        out.append((dcg_val / idcg) if idcg > 0 else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True, choices=["dbpedia", "wikidata"])
    ap.add_argument("--model-type", default="sparqlce", choices=["sparqlce", "sentence-transformer"])
    ap.add_argument("--checkpoint", help="required for --model-type sparqlce")
    ap.add_argument("--model-name", help="HF model id, required for --model-type sentence-transformer")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ablation", default="full", help="tag for the run, e.g. full/no_gnn/baseline-e5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    raw = load_triplets(str(KG_TESTSETS[args.kg]))
    global_corpus, seen_q = build_global_corpus(raw)
    eval_items = build_eval_items(raw, global_corpus, seen_q)
    rel_maps = build_binary_relevance_maps(eval_items, global_corpus)

    if args.model_type == "sparqlce":
        assert args.checkpoint, "--checkpoint required for --model-type sparqlce"
        model = SPARQLEmbedder(args.checkpoint)
        emb_map = build_emb_map_sparqlce(model, global_corpus, batch_size=args.batch_size)
    else:
        from sentence_transformers import SentenceTransformer
        assert args.model_name, "--model-name required for --model-type sentence-transformer"
        model = SentenceTransformer(args.model_name)
        emb_map = build_emb_map_sentence_transformer(model, global_corpus, batch_size=args.batch_size)
    sorted_pools = build_sorted_pools_global(emb_map, eval_items, global_corpus)

    anchors = [a for a, _, _ in eval_items]
    per_anchor = {}
    aggregate = {}
    for k in TOPKS:
        mrr_list = per_anchor_mrr(sorted_pools, eval_items, k)
        ndcg_list = per_anchor_ndcg(sorted_pools, rel_maps, k)
        per_anchor[f"mrr@{k}"] = mrr_list
        per_anchor[f"ndcg@{k}"] = ndcg_list
        aggregate[f"mrr@{k}"] = float(np.mean(mrr_list))
        aggregate[f"ndcg@{k}"] = float(np.mean(ndcg_list))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "kg": args.kg, "model_type": args.model_type,
            "model_id": args.checkpoint or args.model_name, "seed": args.seed,
            "ablation": args.ablation, "anchors": anchors,
            "aggregate": aggregate, "per_anchor": per_anchor,
        }, f, indent=2)
    print(f"Saved per-anchor scores for {len(anchors)} anchors to {out_path}")
    print(f"Aggregate: {json.dumps(aggregate, indent=2)}")


if __name__ == "__main__":
    main()
