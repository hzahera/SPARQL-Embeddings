#!/usr/bin/env python3
"""Per-bin RQ1 evaluation for the stratified split (Option B).

Reuses the corpus/retrieval/metric machinery in
notebooks/evaluation-RQ1/evaluate_rq1.py, but groups anchors by the split's
fixed `lexical_bin` field (NOT the notebook's report_stratified, which re-fits
TF-IDF and drifts). Prints MRR@k / Hits@k per bin for each available method.

Always runs the lexical baselines (TF-IDF, BM25 — no downloads, no checkpoint).
Neural methods run only if their deps/checkpoint are present:
  --st  MODEL_ID   a sentence-transformers model (needs sentence-transformers)
  --sparqlce PATH  a SparqlCE checkpoint (needs torch + sparqlce)

    python scripts/eval_rq1_stratified.py
    python scripts/eval_rq1_stratified.py --st sentence-transformers/all-mpnet-base-v2
    python scripts/eval_rq1_stratified.py --sparqlce path/to/final_model.pt
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "datasets" / "wikidata_testset_RQ1_stratified.json"
TOPKS = [1, 5, 10]
BIN_ORDER = ["<0.3", "0.3-0.5", "0.5-0.8", ">=0.8"]

# --- pure helpers, identical logic to evaluate_rq1.py (kept torch-free here so
# the lexical baselines run without torch/sparqlce/sentence-transformers) ---

def _get_q(x):
    if isinstance(x, dict):
        return str(x.get("query") or x.get("sparql") or "").strip()
    return str(x or "").strip()


def _anchor(item):
    return _get_q(item.get("anchor_query") or item.get("query"))


def build_global_corpus(raw):
    corpus, seen = [], set()
    def add(q):
        if q and q not in seen:
            seen.add(q); corpus.append(q)
    for it in raw:
        a = _anchor(it)
        if not a:
            continue
        add(a)
        for p in (it.get("positive_queries") or []):
            add(_get_q(p))
        for n in (it.get("negative_queries") or []):
            add(_get_q(n))
    return corpus, seen


def build_eval_items(raw, corpus, seen):
    items = []
    for it in raw:
        a = _anchor(it)
        if not a or a not in seen:
            continue
        pos = {_get_q(p) for p in (it.get("positive_queries") or [])
               if _get_q(p) and _get_q(p) in seen and _get_q(p) != a}
        if pos:
            items.append((a, corpus, pos))
    return items


def avg_mrr_at_k(pools, items, k):
    s = 0.0
    for pool, (_, _, pos) in zip(pools, items):
        for rank, q in enumerate(pool[:k], 1):
            if q in pos:
                s += 1.0 / rank
                break
    return s / max(len(items), 1)


def avg_hits_at_k(pools, items, k):
    h = sum(any(q in pos for q in pool[:k]) for pool, (_, _, pos) in zip(pools, items))
    return h / max(len(items), 1)


def sorted_pools_matrix(scores_fn, items, corpus):
    a2i = {q: i for i, q in enumerate(corpus)}
    pools = []
    for a, _, _ in items:
        s = scores_fn(a).copy()
        if a in a2i:
            s[a2i[a]] = -np.inf
        pools.append([corpus[i] for i in np.argsort(-s)])
    return pools


def load_split_and_bins():
    raw = json.load(open(SPLIT))
    corpus, seen = build_global_corpus(raw)
    items = build_eval_items(raw, corpus, seen)
    bin_by_anchor = {_anchor(r): r["lexical_bin"] for r in raw}
    item_bins = [bin_by_anchor.get(a) for (a, _, _) in items]
    return raw, corpus, items, item_bins


def per_bin_report(name, sorted_pools, items, item_bins):
    print(f"\n{'=' * 68}\n{name}\n{'=' * 68}")
    header = f"{'bin':>9} {'n':>4} " + " ".join(f"MRR@{k:<3}" for k in TOPKS) + \
             "  " + " ".join(f"Hit@{k:<3}" for k in TOPKS)
    print(header)
    print("-" * len(header))
    groups = defaultdict(list)
    for sp, it, b in zip(sorted_pools, items, item_bins):
        groups[b].append((sp, it))
    for b in BIN_ORDER + ["ALL"]:
        if b == "ALL":
            pairs = [(sp, it) for sp, it, _ in zip(sorted_pools, items, item_bins)]
        else:
            pairs = groups.get(b, [])
        if not pairs:
            continue
        sps = [p for p, _ in pairs]
        its = [i for _, i in pairs]
        mrr = [avg_mrr_at_k(sps, its, k) for k in TOPKS]
        hit = [avg_hits_at_k(sps, its, k) for k in TOPKS]
        print(f"{b:>9} {len(pairs):>4} " +
              " ".join(f"{m:7.3f}" for m in mrr) + "  " +
              " ".join(f"{h:7.3f}" for h in hit))


def run_tfidf(corpus, items, item_bins):
    X = sk_normalize(TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
                     .fit_transform(corpus), norm="l2", axis=1)
    r2 = {q: i for i, q in enumerate(corpus)}
    def scores(a):
        s = (X @ X[r2[a]].T)
        return (s.toarray().ravel() if hasattr(s, "toarray") else np.asarray(s).ravel())
    per_bin_report("TF-IDF", sorted_pools_matrix(scores, items, corpus), items, item_bins)


def run_bm25(corpus, items, item_bins):
    bm25 = BM25Okapi([q.lower().split() for q in corpus])
    def scores(a):
        return np.asarray(bm25.get_scores(a.lower().split()))
    per_bin_report("BM25", sorted_pools_matrix(scores, items, corpus), items, item_bins)


def _emb_scores(emb_map, corpus):
    M = np.vstack([emb_map[q] for q in corpus])
    r2 = {q: i for i, q in enumerate(corpus)}
    return lambda a: M @ emb_map[a], r2


def run_st(model_id, corpus, items, item_bins):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_id)
    emb = {q: e for q, e in zip(corpus, m.encode(corpus, normalize_embeddings=True,
                                                 convert_to_numpy=True))}
    scores, _ = _emb_scores(emb, corpus)
    per_bin_report(f"SentenceTransformer: {model_id}",
                   sorted_pools_matrix(scores, items, corpus), items, item_bins)


def run_sparqlce(ckpt, corpus, items, item_bins):
    from sparqlce.embedder import SPARQLEmbedder
    m = SPARQLEmbedder(ckpt)
    embs = np.asarray(m.encode(corpus))
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    emb = {q: e for q, e in zip(corpus, embs)}
    scores, _ = _emb_scores(emb, corpus)
    per_bin_report(f"SparqlCE: {Path(ckpt).name}",
                   sorted_pools_matrix(scores, items, corpus), items, item_bins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--st", default=None, help="sentence-transformers model id")
    ap.add_argument("--sparqlce", default=None, help="SparqlCE checkpoint path")
    args = ap.parse_args()

    raw, corpus, items, item_bins = load_split_and_bins()
    print(f"split: {SPLIT.name}  anchors={len(items)}  corpus={len(corpus)}")
    print("bins:", {b: item_bins.count(b) for b in BIN_ORDER})

    run_tfidf(corpus, items, item_bins)
    run_bm25(corpus, items, item_bins)
    if args.st:
        run_st(args.st, corpus, items, item_bins)
    if args.sparqlce:
        run_sparqlce(args.sparqlce, corpus, items, item_bins)


if __name__ == "__main__":
    main()
