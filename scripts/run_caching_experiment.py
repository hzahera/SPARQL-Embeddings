#!/usr/bin/env python3
"""Semantic query-cache simulation (paper §Downstream Application, tab:caching).

Clean, efficient standalone runner for DBpedia or Wikidata. Reproduces the
committed protocol from notebooks/caching/:

  * Stream : held-out test-split anchors, seeded shuffle (5 seeds).
  * Cache  : FIFO over the K most-recent *executed* queries (embedding + result
             set). Incoming query -> max cosine to cache. >= theta => HIT
             (return cached result, do NOT insert). < theta => MISS ("execute"
             via oracle, insert; evict oldest if size > K).
  * Grid   : K x theta, mean +/- std over seeds. Headline (K=128, theta=0.8).
  * Metrics: hit rate; result quality = mean answer-set Jaccard(returned cached
             result, true result) over hits.

Story: SparqlCE's result-aware embeddings recognise result-equivalent queries a
lexical/text cache misses -> higher hit rate at equal-or-better result quality
(dominates the hit-rate/quality Pareto frontier).

Efficiency: the oracle answer sets are loaded ONLY for the stream queries
(streamed with ijson for the 6 GB Wikidata file), not the whole file. Answer sets
are stored as hash sets for memory-light Jaccard. Embeddings are computed once
per model and permuted per seed.

Examples
--------
    # lexical baselines only (no downloads, no checkpoint)
    python scripts/run_caching_experiment.py --kg dbpedia

    # full roster
    python scripts/run_caching_experiment.py --kg wikidata \
        --sparqlce checkpoints_b128/wikidata/seed42_full/best_model.pt \
        --st intfloat/e5-base-v2 --st sentence-transformers/all-mpnet-base-v2 \
        --code-llm Qwen/Qwen2.5-Coder-0.5B \
        --e5-finetuned checkpoints/e5_square_wikidata/best_model

Outputs (results/caching/, gitignored): caching_<kg>_grid.csv,
caching_<kg>_headline.csv, caching_<kg>_headline.tex, caching_<kg>_pareto.csv,
and caching_<kg>_pareto.png if matplotlib is available.
"""
import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # so `sparqlce` imports without an editable install
DS = ROOT / "datasets"
OUT_DIR = ROOT / "results" / "caching"

SEEDS = [42, 1, 2, 3, 4]
K_GRID = [32, 64, 128, 256]
THETA_GRID = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
HEADLINE_K, HEADLINE_THETA = 128, 0.8

_norm = lambda s: " ".join(s.split())


# ---------------------------------------------------------------- oracle ----
def _freeze(v):
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, list):
        return tuple(_freeze(x) for x in v)
    return v


def answer_hashes(rows):
    """Memory-light answer set: hashes of canonicalised rows (Jaccard is exact
    up to negligible hash collisions, and only compared within one run)."""
    return {hash(_freeze(r)) for r in rows}


def load_oracle_for(stream_norm, kg):
    """Return {normalized_query: hashset} for ONLY the stream queries.
    DBpedia: small, json.load. Wikidata: 6 GB, streamed with ijson."""
    path = DS / f"result_sets_{kg}.json"
    oracle = {}
    if kg == "dbpedia":
        raw = json.load(open(path))
        for q, rows in raw.items():
            qn = _norm(q)
            if rows and qn in stream_norm:
                oracle[qn] = answer_hashes(rows)
    else:
        import ijson
        with open(path, "rb") as fh:
            for q, rows in ijson.kvitems(fh, ""):
                qn = _norm(q)
                if rows and qn in stream_norm and qn not in oracle:
                    oracle[qn] = answer_hashes(rows)
    return oracle


def jaccard(a, b):
    if not a and not b:
        return 1.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


# ------------------------------------------------------------ embeddings ----
def bm25_matrix(texts, k1=1.5, b=0.75):
    """L2-normalised Okapi-BM25 term-weight vectors (so the same cosine-theta
    cache protocol applies to BM25 as to the embedding models)."""
    tf = CountVectorizer().fit_transform(texts).astype(float).tocoo()
    n = tf.shape[0]
    dfreq = np.bincount(tf.col, minlength=tf.shape[1])
    idf = np.log(1 + (n - dfreq + 0.5) / (dfreq + 0.5))
    dl = np.asarray(csr_matrix(tf).sum(axis=1)).ravel()
    avgdl = dl.mean() if dl.mean() else 1.0
    denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avgdl)
    w = idf[tf.col] * (tf.data * (k1 + 1)) / denom
    return sk_normalize(csr_matrix((w, (tf.row, tf.col)), shape=tf.shape), axis=1)


def embed(model, texts):
    """Return an L2-normalised representation (sparse for lexical, dense else)."""
    if model == "TF-IDF":
        return sk_normalize(TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
                            .fit_transform(texts), axis=1)
    if model == "BM25":
        return bm25_matrix(texts)
    raise ValueError(model)


def embed_st(model_id, texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_id)
    return np.asarray(m.encode(texts, normalize_embeddings=True, convert_to_numpy=True))


def embed_sparqlce(ckpt, texts):
    from sparqlce.embedder import SPARQLEmbedder
    e = np.asarray(SPARQLEmbedder(ckpt).encode(texts))
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


def embed_e5_finetuned(ckpt, texts, batch_size=16, max_length=512):
    """Fine-tuned E5 checkpoint (AutoModel + tokenizer, saved via
    ``backbone.save_pretrained``/``tokenizer.save_pretrained`` in
    notebooks/finetuning/*_e5_contrastive_finetune_square*). Mean-pools over
    non-padding tokens and applies the "query: " E5 prefix, matching training
    (see E5SparqlEmbedder / mean_pool in that notebook)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ckpt)
    mdl = AutoModel.from_pretrained(ckpt).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = ["query: " + t for t in texts[i:i + batch_size]]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(dev)
            h = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            out.append(pooled.cpu().numpy())
    return np.vstack(out)


def embed_code_llm(model_id, texts, batch_size=16, max_length=512):
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32
    ).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(dev)
            h = mdl(**enc).last_hidden_state
            am = enc["attention_mask"]
            if am[:, -1].sum() == am.shape[0]:
                pooled = h[:, -1]
            else:
                pooled = h[torch.arange(h.shape[0]), am.sum(1) - 1]
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            out.append(pooled.cpu().numpy())
    return np.vstack(out)


# ----------------------------------------------------------- simulation ----
def simulate(E, ans, order, K, theta):
    """One FIFO cache pass. E: (n,d) rows (sparse or dense). ans: list of
    hashsets. order: arrival permutation. Returns (hit_rate, result_quality)."""
    sparse = hasattr(E, "toarray")
    dvec = (lambda i: np.asarray(E[i].todense()).ravel()) if sparse else (lambda i: E[i])
    d = E.shape[1]
    idxs = deque()
    mat = np.zeros((0, d), dtype=np.float32)
    hits, qsum, n = 0, 0.0, 0
    for i in order:
        n += 1
        if len(idxs):
            sims = mat @ dvec(i)
            j = int(np.argmax(sims))
            if sims[j] >= theta:
                hits += 1
                qsum += jaccard(ans[idxs[j]], ans[i])
                continue
        idxs.append(i)
        mat = np.vstack([mat, dvec(i).astype(np.float32)[None, :]])
        if len(idxs) > K:
            idxs.popleft()
            mat = mat[1:]
    return hits / n, (qsum / hits if hits else 0.0)


def run_model(name, E, ans, n):
    """Full K x theta x seed grid -> {(K,theta): (hr_mean,hr_std,q_mean,q_std)}."""
    perms = [np.random.default_rng(s).permutation(n) for s in SEEDS]
    grid = {}
    for K in K_GRID:
        for th in THETA_GRID:
            hrs, qs = [], []
            for order in perms:
                hr, q = simulate(E, ans, order, K, th)
                hrs.append(hr); qs.append(q)
            grid[(K, th)] = (np.mean(hrs), np.std(hrs), np.mean(qs), np.std(qs))
    return grid


# ---------------------------------------------------------------- driver ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True, choices=["dbpedia", "wikidata"])
    ap.add_argument("--sparqlce", default=None)
    ap.add_argument("--st", action="append", default=[], metavar="MODEL_ID")
    ap.add_argument("--code-llm", action="append", default=[], metavar="MODEL_ID")
    ap.add_argument("--e5-finetuned", default=None, metavar="CKPT_DIR",
                     help="path to a fine-tuned E5 checkpoint dir (e.g. "
                          "checkpoints/e5_square_<kg>/best_model)")
    ap.add_argument("--no-lexical", action="store_true")
    ap.add_argument("--min-answer-size", type=int, default=1, metavar="N",
                     help="drop stream queries whose true answer set has fewer "
                          "than N rows (default 1 = no filtering; singleton "
                          "answer sets make Jaccard quality all-or-nothing)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    trip = json.load(open(DS / f"{args.kg}-triplets-dataset.json"))
    spl = json.load(open(DS / f"splits_{args.kg}.json"))
    # unique held-out test anchors, stable order
    seen, stream = set(), []
    for i in spl["test_idx"]:
        a = trip[i]["anchor_query"]; an = _norm(a)
        if an not in seen:
            seen.add(an); stream.append(a)
    print(f"[{args.kg}] test anchors: {len(stream)}")

    oracle = load_oracle_for({_norm(a) for a in stream}, args.kg)
    stream = [a for a in stream if _norm(a) in oracle]     # oracle-covered only
    ans = [oracle[_norm(a)] for a in stream]
    n = len(stream)
    print(f"[{args.kg}] oracle-covered stream: {n}  "
          f"(answer-set sizes: min {min(len(x) for x in ans)} "
          f"median {int(np.median([len(x) for x in ans]))} max {max(len(x) for x in ans)})")
    if n < 10:
        raise SystemExit("Too few oracle-covered anchors — check result_sets file.")

    if args.min_answer_size > 1:
        kept = [(a, x) for a, x in zip(stream, ans) if len(x) >= args.min_answer_size]
        stream, ans = [a for a, _ in kept], [x for _, x in kept]
        n = len(stream)
        print(f"[{args.kg}] after --min-answer-size {args.min_answer_size}: {n} queries remain")
        if n < 10:
            raise SystemExit("Too few queries left after --min-answer-size filter.")

    # Text-based baselines need Wikidata's opaque P/Q ids expanded to English
    # labels to see the same semantic content SparqlCE's graph builder resolves
    # internally (sparqlce.sparql_utils.expand_wikidata_labels docstring); a
    # no-op on DBpedia. SparqlCE keeps the raw SPARQL -- the expanded text is
    # not valid SPARQL syntax, only a plain-text stand-in for tokenizers.
    if args.kg == "wikidata":
        from sparqlce.sparql_utils import expand_wikidata_labels
        text_stream = [expand_wikidata_labels(a) for a in stream]
    else:
        text_stream = stream

    # model roster
    jobs = []
    if not args.no_lexical:
        jobs += [("TF-IDF", lambda: embed("TF-IDF", text_stream)),
                 ("BM25", lambda: embed("BM25", text_stream))]
    for mid in args.st:
        short = mid.split("/")[-1]
        jobs.append((short, (lambda m=mid: embed_st(m, text_stream))))
    for mid in getattr(args, "code_llm"):
        jobs.append((mid.split("/")[-1], (lambda m=mid: embed_code_llm(m, text_stream))))
    if args.sparqlce:
        jobs.append(("SparqlCE", lambda: embed_sparqlce(args.sparqlce, stream)))
    if args.e5_finetuned:
        jobs.append(("E5-ft", (lambda p=args.e5_finetuned: embed_e5_finetuned(p, text_stream))))

    grids = {}
    for name, emb_fn in jobs:
        print(f"  embedding + simulating: {name} ...", flush=True)
        try:
            E = emb_fn()
        except Exception as e:
            print(f"    skip {name}: {e}")
            continue
        grids[name] = run_model(name, E, ans, n)

    # ---- outputs ----
    kg = args.kg
    # full grid
    with open(OUT_DIR / f"caching_{kg}_grid.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "K", "theta", "hit_rate", "hit_std", "quality", "quality_std"])
        for name, g in grids.items():
            for (K, th), (hr, hs, q, qs) in g.items():
                w.writerow([name, K, th, f"{hr:.4f}", f"{hs:.4f}", f"{q:.4f}", f"{qs:.4f}"])

    # headline table (K=128, theta=0.8), sorted by hit rate
    rows = sorted(((name, *g[(HEADLINE_K, HEADLINE_THETA)]) for name, g in grids.items()),
                  key=lambda r: -r[1])
    print(f"\n=== HEADLINE  (K={HEADLINE_K}, theta={HEADLINE_THETA})  [{kg}] ===")
    print(f"{'model':>16} {'hit_rate':>16} {'result_quality':>18}")
    with open(OUT_DIR / f"caching_{kg}_headline.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "hit_rate", "hit_std", "quality", "quality_std"])
        for name, hr, hs, q, qs in rows:
            print(f"{name:>16} {hr:>7.3f} +/- {hs:5.3f} {q:>10.3f} +/- {qs:5.3f}")
            w.writerow([name, f"{hr:.4f}", f"{hs:.4f}", f"{q:.4f}", f"{qs:.4f}"])

    # LaTeX headline
    with open(OUT_DIR / f"caching_{kg}_headline.tex", "w") as f:
        best_hr = max(r[1] for r in rows); best_q = max(r[3] for r in rows)
        f.write("\\begin{tabular}{lcc}\n\\toprule\n"
                f"Model & Hit rate & Result quality \\\\\n\\midrule\n")
        for name, hr, hs, q, qs in rows:
            nm = "\\textbf{SparqlCE (ours)}" if name == "SparqlCE" else name
            hrc = f"\\textbf{{{hr:.3f}}}" if abs(hr-best_hr) < 1e-9 else f"{hr:.3f}"
            qc = f"\\textbf{{{q:.3f}}}" if abs(q-best_q) < 1e-9 else f"{q:.3f}"
            f.write(f"{nm} & {hrc}\\,$\\pm$\\,{hs:.3f} & {qc}\\,$\\pm$\\,{qs:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # Pareto (theta sweep at headline K)
    with open(OUT_DIR / f"caching_{kg}_pareto.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "theta", "hit_rate", "quality"])
        for name, g in grids.items():
            for th in THETA_GRID:
                hr, _, q, _ = g[(HEADLINE_K, th)]
                w.writerow([name, th, f"{hr:.4f}", f"{q:.4f}"])
    _pareto_plot(grids, kg)
    print(f"\nwrote grid/headline/pareto to {OUT_DIR}")


def _pareto_plot(grids, kg):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, g in grids.items():
        hr = [g[(HEADLINE_K, th)][0] for th in THETA_GRID]
        q = [g[(HEADLINE_K, th)][2] for th in THETA_GRID]
        lw = 2.6 if name == "SparqlCE" else 1.3
        z = 5 if name == "SparqlCE" else 2
        ax.plot(hr, q, marker="o", ms=4, lw=lw, zorder=z, label=name)
    ax.set_xlabel("Hit rate"); ax.set_ylabel("Result quality (answer-set Jaccard)")
    ax.set_title(f"Semantic cache Pareto — {kg} (K={HEADLINE_K}, θ sweep)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT_DIR / f"caching_{kg}_pareto.png", dpi=150)


if __name__ == "__main__":
    main()
