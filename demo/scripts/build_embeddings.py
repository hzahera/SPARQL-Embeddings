#!/usr/bin/env python3
"""Build the 2-D embedding maps shown in demo/embeddings.html.

For every test query of SQuARE (473 DBpedia, 968 Wikidata) it computes one or more
embedding "spaces", reduces each to 2-D with t-SNE, and stores the nearest neighbour
of every query in the full-dimensional space (cosine).

Spaces:
  tfidf       text-based TF-IDF vectors (fitted on the training split)          [always]
  target      answer-overlap layout from the complete test labels (1 - J)       [always]
  zeroshot    pretrained encoder, before fine-tuning                            [--model]
  finetuned   the same encoder after results-aware fine-tuning                  [--model]

The two encoder spaces need torch + transformers, the repository's src/ package and a
trained checkpoint (results/raw/<graph>/<model>/tuned/seed_<seed>/best_model).

Usage (from the repository root):
    python demo/scripts/build_embeddings.py                                 # tfidf + target only (CPU)
    python demo/scripts/build_embeddings.py --model e5-base-v2 --seed 42    # + real encoders (GPU recommended)
    python demo/scripts/bundle_demo.py                                      # rebuild docs/*.html
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_demo_data import strip_prefixes, toks, ws  # noqa: E402


def tsne_2d(X: np.ndarray, metric: str = "cosine", seed: int = 0) -> np.ndarray:
    from sklearn.manifold import TSNE

    perplexity = min(30, max(5, (len(X) - 1) // 4))
    if metric == "precomputed":
        Y = TSNE(n_components=2, metric="precomputed", init="random", perplexity=perplexity,
                 random_state=seed).fit_transform(X)
    else:
        Y = TSNE(n_components=2, metric=metric, init="pca", perplexity=perplexity,
                 random_state=seed).fit_transform(X)
    Y = Y - Y.mean(0)
    Y = Y / (np.abs(Y).max() or 1.0)
    return np.round(Y, 4)


def nearest(sim: np.ndarray) -> list[int]:
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    return [int(i) for i in s.argmax(1)]


def load_graph(repo: Path, g: str):
    d = repo / "datasets" / f"{g}-splits"
    rq3 = json.loads((d / f"{g}-test-rq3.json").read_text())
    train = json.loads((d / f"{g}-train.json").read_text())
    qs = [x["query"] for x in rq3["queries"]]
    n_ans = [x.get("n_answers") for x in rq3["queries"]]
    J = {}
    for a, c, j in rq3["positive_pairs"] + rq3["negative_pairs"]:
        J[(a, c)] = J[(c, a)] = float(j)
    train_q = list(dict.fromkeys(
        ws(c) for r in train
        for c in [r["anchor_query"]] + [x["query"] for x in r["positive_queries"] + r["negative_queries"]]))
    return qs, n_ans, J, train_q


def components(n: int, J: dict) -> list[int]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, c), j in J.items():
        if j >= 0.8:
            parent[find(a)] = find(c)
    roots = [find(i) for i in range(n)]
    sizes = Counter(roots)
    # rank components by size; groups of size 1 get -1
    order = {r: k for k, (r, s) in enumerate(sizes.most_common()) if s > 1}
    return [order.get(r, -1) for r in roots]


def tfidf_space(qs, train_q):
    from scipy.sparse import csr_matrix

    train_toks = [toks(q) for q in train_q]
    n = len(train_toks)
    df = Counter(t for d in train_toks for t in set(d))
    vocab = {}
    rows, cols, vals = [], [], []
    for i, q in enumerate(qs):
        c = Counter(toks(ws(q)))
        v = {t: (1 + math.log(k)) * (math.log((1 + n) / (1 + df.get(t, 0))) + 1) for t, k in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        for t, x in v.items():
            rows.append(i)
            cols.append(vocab.setdefault(t, len(vocab)))
            vals.append(x / norm)
    X = csr_matrix((vals, (rows, cols)), shape=(len(qs), len(vocab)))
    sim = (X @ X.T).toarray()
    from sklearn.decomposition import TruncatedSVD

    Xr = TruncatedSVD(n_components=min(100, X.shape[1] - 1), random_state=0).fit_transform(X)
    return tsne_2d(Xr), nearest(sim), sim


def target_space(n, J):
    # distance = 1 - J; pairs missing from the complete labels have 0.1 < J < 0.8 -> use 0.55
    D = np.full((n, n), 0.55)
    for (a, c), j in J.items():
        D[a, c] = 1.0 - j
    np.fill_diagonal(D, 0.0)
    sim = 1.0 - D
    return tsne_2d(D, metric="precomputed"), nearest(sim + np.random.default_rng(0).random((n, n)) * 1e-6)


def encoder_spaces(repo: Path, g: str, qs: list[str], model_key: str, seed: int, device: str):
    import torch  # noqa: F401
    import yaml

    sys.path.insert(0, str(repo / "src"))
    from result_aware.model import HFEmbeddingModel

    cfg = None
    try:  # prefer the repository's own config loader if present
        from result_aware import config as rcfg

        for fn in ("load_model_config", "get_model_config", "model_config"):
            if hasattr(rcfg, fn):
                cfg = getattr(rcfg, fn)(model_key)
                break
    except Exception:
        cfg = None
    if cfg is None:
        cfg = yaml.safe_load((repo / "configs" / "models" / f"{model_key}.yaml").read_text())
    ckpt = repo / "results" / "raw" / g / model_key / "tuned" / f"seed_{seed}" / "best_model"
    if not ckpt.exists():
        raise FileNotFoundError(f"fine-tuned checkpoint not found: {ckpt}")

    out = {}
    for kind, path in (("zeroshot", None), ("finetuned", ckpt)):
        m = HFEmbeddingModel(cfg, device=device, checkpoint_path=path)
        E = m.encode(qs, batch_size=32, show_progress=True).float().numpy()
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        sim = E @ E.T
        out[kind] = (tsne_2d(E), nearest(sim), sim)
        del m
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    return cfg.get("name") or cfg.get("display_name") or model_key, out


def hits1(nn, J):
    # over queries that have at least one high-overlap query in the test split
    has = {a for (a, c), j in J.items() if j >= 0.8}
    ok = [J.get((i, nn[i]), 0.0) >= 0.8 for i in sorted(has)]
    return round(sum(ok) / max(1, len(ok)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="demo/data/embeddings.js")
    ap.add_argument("--graphs", nargs="+", default=["dbpedia", "wikidata"])
    ap.add_argument("--model", default=None, help="model key in configs/models, e.g. e5-base-v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    out_path = Path(args.out)

    data = {}
    for g in args.graphs:
        qs, n_ans, J, train_q = load_graph(repo, g)
        n = len(qs)
        groups = components(n, J)
        spaces = {}
        xy, nn, _ = tfidf_space(qs, train_q)
        spaces["tfidf"] = {"label": "Text-based (TF-IDF)", "kind": "text", "xy": xy.tolist(), "nn": nn}
        xy, nn = target_space(n, J)
        spaces["target"] = {"label": "Answer overlap (target)", "kind": "target", "xy": xy.tolist(), "nn": nn}
        if args.model:
            name, enc = encoder_spaces(repo, g, qs, args.model, args.seed, args.device)
            spaces["zeroshot"] = {"label": f"{name}, zero-shot", "kind": "text",
                                  "xy": enc["zeroshot"][0].tolist(), "nn": enc["zeroshot"][1]}
            spaces["finetuned"] = {"label": f"{name}, results-aware fine-tuned", "kind": "ours",
                                   "xy": enc["finetuned"][0].tolist(), "nn": enc["finetuned"][1]}
        for k, s in spaces.items():
            s["hits1"] = hits1(s["nn"], J)
            print(f"{g:9s} {k:10s} nearest neighbour shares answers (J>=0.8): {s['hits1']:.3f}")
        pos = {}
        for (a, c), j in J.items():
            if j >= 0.8:
                pos.setdefault(a, []).append(c)
        data[g] = {
            "queries": [{"q": " ".join(strip_prefixes(q).split()), "n": n_ans[i], "g": groups[i]}
                        for i, q in enumerate(qs)],
            "pos": {str(k): sorted(v) for k, v in pos.items()},
            "spaces": spaces,
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("// Generated by demo/scripts/build_embeddings.py from the SQuARE test splits.\n"
                        "window.EMB = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print("wrote", out_path, f"{out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
