#!/usr/bin/env python3
"""Compare answer-label and syntax-label fine-tuning on the same encoders and seeds.

For every graph, encoder and seed, both checkpoints rank all test queries for every test
anchor, as in the paper:
  stored   : relevance = stored J (unstored = 0)          -> Hits@1, MRR@5, nDCG@5   (Table 2)
  complete : relevance = complete J from *-test-rq3.json  -> Hits@1, MRR,   nDCG@10  (Table 3)
             (J >= 0.8 relevant; lower J counts as non-relevant, as in nonneural.py)
Per-anchor scores are averaged over seeds, then over encoders. Intervals are paired 95%
percentile bootstrap intervals over test anchors (answer minus syntax).

Usage (from the repository root, GPU recommended):
  python scripts/syntax_control/eval_syntax_control.py
Outputs:
  scripts/syntax_control/control_results.json
  scripts/syntax_control/control_latex.txt   (table rows and paragraph numbers)
Per-run scores are cached in scripts/syntax_control/cache/ so reruns are fast.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))
import audit_splits as au  # noqa: E402

from result_aware.config import load_model_config  # noqa: E402
from result_aware.metrics import hits_at_k, mrr_at_k, ndcg_at_k  # noqa: E402
from result_aware.model import HFEmbeddingModel  # noqa: E402

ws = au.ws
METRICS = {
    "stored": ["hits@1", "mrr@5", "ndcg@5"],
    "complete": ["hits@1", "mrr", "ndcg@10"],
}
CONDITIONS = ("answer", "syntax")


def load_graph(root: Path, g: str):
    d = root / "datasets" / f"{g}-splits"
    test = json.load(open(d / f"{g}-test.json", encoding="utf-8"))
    rq3 = json.load(open(d / f"{g}-test-rq3.json", encoding="utf-8"))
    queries = [x["query"] for x in rq3["queries"]]
    idx = {ws(q): i for i, q in enumerate(queries)}
    full: dict[tuple[int, int], float] = {}
    for a, c, j in rq3["positive_pairs"] + rq3["negative_pairs"]:
        full[(a, c)] = full[(c, a)] = float(j)
    anchors = []
    for r in test:
        a = idx[ws(r["anchor_query"])]
        stored = {idx[ws(x["query"])]: float(x["answer_jaccard"])
                  for x in r["positive_queries"] + r["negative_queries"]}
        anchors.append((a, stored))
    return queries, full, anchors


def per_anchor_scores(emb: torch.Tensor, full, anchors) -> dict[str, dict[str, list[float]]]:
    emb = torch.nn.functional.normalize(emb.float(), dim=-1)
    sims = (emb @ emb.T).numpy()
    n = sims.shape[0]
    out = {p: {m: [] for m in ms} for p, ms in METRICS.items()}
    for a, stored in anchors:
        cands = np.array([c for c in range(n) if c != a])
        order = cands[np.argsort(-sims[a, cands], kind="stable")]
        # stored labels
        rels = [stored.get(int(c), 0.0) for c in order]
        binr = [x >= 0.8 for x in rels]
        out["stored"]["hits@1"].append(hits_at_k(binr, 1))
        out["stored"]["mrr@5"].append(mrr_at_k(binr, 5))
        out["stored"]["ndcg@5"].append(ndcg_at_k(rels, 5))
        # complete labels
        rels = [full.get((a, int(c)), 0.0) for c in order]
        rels = [x if x >= 0.8 else 0.0 for x in rels]
        binr = [x >= 0.8 for x in rels]
        out["complete"]["hits@1"].append(hits_at_k(binr, 1))
        out["complete"]["mrr"].append(mrr_at_k(binr, len(binr)))
        out["complete"]["ndcg@10"].append(ndcg_at_k(rels, 10))
    return out


def run_scores(root, results_dir, g, model_key, seed, queries, full, anchors, device, cache_dir, cond):
    cache = cache_dir / f"{g}__{model_key}__seed{seed}__{cond}.json"
    if cache.exists():
        return json.load(open(cache))
    ckpt = results_dir / "raw" / g / model_key / "tuned" / f"seed_{seed}" / "best_model"
    if not ckpt.exists():
        raise SystemExit(f"Missing checkpoint for {cond}: {ckpt}")
    cfg = load_model_config(model_key, root / "configs")
    model = HFEmbeddingModel(cfg, device=device, checkpoint_path=ckpt)
    emb = model.encode(queries, batch_size=64, show_progress=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    scores = per_anchor_scores(emb, full, anchors)
    cache.parent.mkdir(parents=True, exist_ok=True)
    json.dump(scores, open(cache, "w"))
    return scores


def bootstrap(diff: np.ndarray, reps: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(reps)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--answer-results", default="results", help="output_dir of the main runs")
    ap.add_argument("--syntax-results", default="results_syntax_control")
    ap.add_argument("--models", nargs="+", default=["e5-base-v2", "qwen3-embedding", "codebert"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--graphs", nargs="+", default=["dbpedia", "wikidata"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()

    root = Path(args.root)
    res_dir = {"answer": root / args.answer_results, "syntax": root / args.syntax_results}
    cache_dir = root / "scripts" / "syntax_control" / "cache"
    report: dict = {"models": args.models, "seeds": args.seeds, "graphs": {}}

    for g in args.graphs:
        queries, full, anchors = load_graph(root, g)
        # per condition -> protocol -> metric -> [models x anchors]
        agg = {c: {p: {m: [] for m in ms} for p, ms in METRICS.items()} for c in CONDITIONS}
        per_model = {}
        for model_key in args.models:
            per_model[model_key] = {}
            for cond in CONDITIONS:
                runs = [run_scores(root, res_dir[cond], g, model_key, s, queries, full, anchors,
                                   args.device, cache_dir, cond) for s in args.seeds]
                per_model[model_key][cond] = {}
                for p, ms in METRICS.items():
                    per_model[model_key][cond][p] = {}
                    for m in ms:
                        seed_mean = np.mean([r[p][m] for r in runs], axis=0)  # per anchor
                        agg[cond][p][m].append(seed_mean)
                        per_model[model_key][cond][p][m] = round(float(seed_mean.mean()), 4)
            # per-model Hits@1 difference under complete labels
            a = np.mean([run_scores(root, res_dir["answer"], g, model_key, s, queries, full, anchors,
                                    args.device, cache_dir, "answer")["complete"]["hits@1"] for s in args.seeds], axis=0)
            b = np.mean([run_scores(root, res_dir["syntax"], g, model_key, s, queries, full, anchors,
                                    args.device, cache_dir, "syntax")["complete"]["hits@1"] for s in args.seeds], axis=0)
            lo, hi = bootstrap(a - b, args.bootstrap)
            per_model[model_key]["complete_hits@1_diff"] = {
                "mean": round(float((a - b).mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}

        gr = {"anchors": len(anchors), "test_queries": len(queries), "per_model": per_model, "mean": {}}
        for p, ms in METRICS.items():
            gr["mean"][p] = {}
            for m in ms:
                A = np.mean(agg["answer"][p][m], axis=0)
                B = np.mean(agg["syntax"][p][m], axis=0)
                lo, hi = bootstrap(A - B, args.bootstrap)
                gr["mean"][p][m] = {
                    "answer": round(float(A.mean()), 4),
                    "syntax": round(float(B.mean()), 4),
                    "diff": round(float((A - B).mean()), 4),
                    "ci95": [round(lo, 4), round(hi, 4)],
                }
        report["graphs"][g] = gr

    out_dir = root / "scripts" / "syntax_control"
    json.dump(report, open(out_dir / "control_results.json", "w"), indent=2)

    def f(x):
        return f"{x:.2f}"

    lines = [f"% Encoders: {', '.join(args.models)}; seeds: {args.seeds}"]
    for p, label, tab in (("stored", "stored labels", "Tables/compact/retrieval.tex"),
                          ("complete", "complete labels", "Tables/compact/fullcollection.tex")):
        lines.append(f"% {tab} ({label}), columns: DBpedia {METRICS[p]} & Wikidata {METRICS[p]}")
        for cond, name in (("answer", "Answer labels (3 encoders)"), ("syntax", "Syntax labels (3 encoders)")):
            cells = []
            for g in ("dbpedia", "wikidata"):
                if g in report["graphs"]:
                    cells += [f(report["graphs"][g]["mean"][p][m][cond]) for m in METRICS[p]]
                else:
                    cells += ["--"] * 3
            lines.append(f"{name} & " + " & ".join(cells[:3]) + " & & " + " & ".join(cells[3:]) + r"\\")
    lines.append("% Paragraph numbers (complete labels, Hits@1): syntax, answer, diff [CI]")
    for g, gr in report["graphs"].items():
        h = gr["mean"]["complete"]["hits@1"]
        lines.append(f"% {g}: syntax {f(h['syntax'])}, answer {f(h['answer'])}, "
                     f"diff {h['diff']:+.2f} [{h['ci95'][0]:.2f}, {h['ci95'][1]:.2f}]")
    (out_dir / "control_latex.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
