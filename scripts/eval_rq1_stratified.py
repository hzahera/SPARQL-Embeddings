#!/usr/bin/env python3
"""Per-bin RQ1 evaluation for the stratified split (Option B).

Reuses the corpus/retrieval/metric machinery in
notebooks/evaluation-RQ1/evaluate_rq1.py, but groups anchors by the split's
fixed `lexical_bin` field (NOT the notebook's report_stratified, which re-fits
TF-IDF and drifts). Prints MRR@k / Hits@k per bin for each available method.

Always runs the lexical baselines (TF-IDF, BM25 — no downloads, no checkpoint)
unless --no-lexical. Neural methods run only if their deps are present:
  --st MODEL_ID     mean-pooled encoder / SentenceTransformer (repeatable)
  --code-llm MODEL_ID  decoder code-LLM, last-token pooled (repeatable)
  --sparqlce PATH   a SparqlCE checkpoint (needs torch + sparqlce)

    python scripts/eval_rq1_stratified.py
    python scripts/eval_rq1_stratified.py --st sentence-transformers/all-mpnet-base-v2

    # full remaining paper roster in one run:
    python scripts/eval_rq1_stratified.py \
      --st intfloat/e5-base-v2 \
      --st sentence-transformers/all-MiniLM-L6-v2 \
      --st microsoft/codebert-base \
      --st microsoft/graphcodebert-base \
      --st razent/spbert-mlm-wso-base \
      --code-llm Qwen/Qwen2.5-Coder-0.5B \
      --sparqlce checkpoints_b128/wikidata/seed42_full/best_model.pt
"""
import argparse
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # so `import sparqlce` (repo root) works from scripts/
SPLIT = ROOT / "datasets" / "wikidata-testsplit-RQ1.json"
TOPKS = [1, 3, 5, 10]   # @3/@5/@10 reported in the paper (matches the DBpedia table)
BIN_ORDER = ["<0.3", "0.3-0.5", "0.5-0.8", ">=0.8"]

_WD_ID_RE = re.compile(r'\b[PQ]\d+\b')


def _load_label_expander():
    """Import expand_wikidata_labels without importing the sparqlce package.

    `sparqlce/__init__.py` imports torch; sparql_utils does not. Loading the
    submodule by path keeps the lexical rows runnable on a torch-free machine.
    """
    path = ROOT / "sparqlce" / "sparql_utils.py"
    spec = importlib.util.spec_from_file_location("_sparqlce_sparql_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.expand_wikidata_labels


def _label_coverage(corpus, expand):
    """(distinct P/Q ids in the corpus, how many the label map resolves)."""
    ids = set()
    for q in corpus:
        ids.update(_WD_ID_RE.findall(q))
    # An id is resolved iff expanding a probe containing it changes the string.
    resolved = sum(1 for i in ids if expand(f"wdt:{i}") != f"wdt:{i}")
    return len(ids), resolved


def _is_wikidata(corpus):
    """True if the corpus carries opaque P/Q ids that label expansion targets."""
    if os.environ.get("SPARQLCE_NO_WIKIDATA_LABELS", "").strip().lower() \
            in ("1", "true", "yes", "on"):
        return False
    return any(_WD_ID_RE.search(q) for q in corpus[:50])


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


def _dcg(rels):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def avg_ndcg_at_k(pools, items, k):
    """Binary-relevance nDCG@k, identical to evaluate_rq1.avg_ndcg_binary:
    DCG over top-k (rel 1 if positive) / ideal DCG of min(#pos, k) ones."""
    tot = 0.0
    for pool, (_, _, pos) in zip(pools, items):
        rels = [1.0 if q in pos else 0.0 for q in pool[:k]]
        idcg = _dcg([1.0] * min(len(pos), k))
        tot += (_dcg(rels) / idcg) if idcg > 0 else 0.0
    return tot / max(len(items), 1)


def sorted_pools_matrix(scores_fn, items, corpus):
    a2i = {q: i for i, q in enumerate(corpus)}
    pools = []
    for a, _, _ in items:
        s = scores_fn(a).copy()
        if a in a2i:
            s[a2i[a]] = -np.inf
        # kind="stable": ties MUST break by corpus position, which is fixed by
        # the split file. Default argsort is introsort -- tie order then depends
        # on the numpy build, so the same split scores differently on two
        # machines. This is not a corner case for the lexical rows: on
        # dbpedia-testsplit-RQ1, 219/219 anchors (BM25) and 218/219 (TF-IDF)
        # have tied scores inside the top-10, and the drift is worth ~0.04
        # MRR@10 on BM25 -- larger than several margins the paper reports.
        pools.append([corpus[i] for i in np.argsort(-s, kind="stable")])
    return pools


def load_split_and_bins(split_path=SPLIT):
    raw = json.load(open(split_path))
    corpus, seen = build_global_corpus(raw)
    items = build_eval_items(raw, corpus, seen)
    bin_by_anchor = {_anchor(r): r.get("lexical_bin", "all") for r in raw}
    item_bins = [bin_by_anchor.get(a) for (a, _, _) in items]
    return raw, corpus, items, item_bins


def stratify_by_lexical(corpus, text_corpus, items):
    """Assign each anchor a lexical bin from its mean TF-IDF cosine (over the
    label-EXPANDED text -- the same text baselines read) to its positives. Bins
    match BIN_ORDER so per_bin_report groups them. Computed on the fly so a plain
    split with no `lexical_bin` field can still be reported per-stratum."""
    X = sk_normalize(TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
                     .fit_transform(text_corpus), norm="l2", axis=1)
    idx = {q: i for i, q in enumerate(corpus)}
    bins = []
    for a, _, pos in items:
        cs = [float(X[idx[a]].multiply(X[idx[p]]).sum())
              for p in pos if p in idx]
        m = float(np.mean(cs)) if cs else 0.0
        bins.append("<0.3" if m < 0.3 else "0.3-0.5" if m < 0.5
                    else "0.5-0.8" if m < 0.8 else ">=0.8")
    return bins


def per_bin_report(name, sorted_pools, items, item_bins):
    print(f"\n{'=' * 68}\n{name}\n{'=' * 68}")
    header = f"{'bin':>9} {'n':>4} " + " ".join(f"MRR@{k:<3}" for k in TOPKS) + \
             "  " + " ".join(f"Hit@{k:<3}" for k in TOPKS) + \
             "  " + " ".join(f"nDCG@{k:<2}" for k in TOPKS)
    print(header)
    print("-" * len(header))
    groups = defaultdict(list)
    for sp, it, b in zip(sorted_pools, items, item_bins):
        groups[b].append((sp, it))
    present = [b for b in BIN_ORDER if b in groups] + \
              [b for b in groups if b not in BIN_ORDER]
    for b in present + ["ALL"]:
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
        ndcg = [avg_ndcg_at_k(sps, its, k) for k in TOPKS]
        print(f"{b:>9} {len(pairs):>4} " +
              " ".join(f"{m:7.3f}" for m in mrr) + "  " +
              " ".join(f"{h:7.3f}" for h in hit) + "  " +
              " ".join(f"{d:7.3f}" for d in ndcg))


def run_tfidf(corpus, text_corpus, items, item_bins):
    X = sk_normalize(TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
                     .fit_transform(text_corpus), norm="l2", axis=1)
    r2 = {q: i for i, q in enumerate(corpus)}
    def scores(a):
        s = (X @ X[r2[a]].T)
        return (s.toarray().ravel() if hasattr(s, "toarray") else np.asarray(s).ravel())
    per_bin_report("TF-IDF", sorted_pools_matrix(scores, items, corpus), items, item_bins)


def run_bm25(corpus, text_corpus, items, item_bins):
    bm25 = BM25Okapi([q.lower().split() for q in text_corpus])
    xform = dict(zip(corpus, text_corpus))
    def scores(a):
        return np.asarray(bm25.get_scores(xform[a].lower().split()))
    per_bin_report("BM25", sorted_pools_matrix(scores, items, corpus), items, item_bins)


def _emb_scores(emb_map, corpus):
    M = np.vstack([emb_map[q] for q in corpus])
    r2 = {q: i for i, q in enumerate(corpus)}
    return lambda a: M @ emb_map[a], r2


E5_PREFIX = "query: "


def run_st(model_id, corpus, text_corpus, items, item_bins):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_id)
    # E5 is trained with a "query: " prefix on every input -- both the zero-shot
    # checkpoint and the SQuARE fine-tune (finetune_e5_baseline.py applies it to
    # anchors, positives and negatives alike). Encoding without it here is a
    # train/test mismatch that costs the E5 rows accuracy for no methodological
    # reason. Matches eval_rq1_export.py's --text-prefix auto.
    texts = text_corpus
    if "e5" in str(model_id).lower():
        texts = [E5_PREFIX + t for t in text_corpus]
        print(f"  [parity] prepending {E5_PREFIX!r} for {model_id}")
    encoded = m.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    emb = {q: e for q, e in zip(corpus, encoded)}
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


def _causal_last_token_pool(last_hidden, attn):
    import torch
    if attn[:, -1].sum() == attn.shape[0]:      # left padding
        return last_hidden[:, -1]
    idx = attn.sum(dim=1) - 1
    return last_hidden[torch.arange(last_hidden.shape[0]), idx]


def run_code_llm(model_id, corpus, text_corpus, items, item_bins, batch_size=16, max_length=512):
    """Decoder code-LLM (e.g. Qwen2.5-Coder) as an embedder via last-token pooling
    — same recipe as evaluate_rq1.build_emb_map_code_llm."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    emb = {}
    with torch.no_grad():
        for i in range(0, len(corpus), batch_size):
            keys = corpus[i:i + batch_size]
            batch = text_corpus[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt").to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            pooled = _causal_last_token_pool(out.last_hidden_state, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            for q, e in zip(keys, pooled.cpu().numpy()):
                emb[q] = e
    scores, _ = _emb_scores(emb, corpus)
    per_bin_report(f"CodeLLM: {model_id}",
                   sorted_pools_matrix(scores, items, corpus), items, item_bins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--st", action="append", default=[], metavar="MODEL_ID",
                    help="sentence-transformers / mean-pooled encoder id (repeatable)")
    ap.add_argument("--code-llm", action="append", default=[], metavar="MODEL_ID",
                    help="decoder code-LLM id, last-token pooled (repeatable)")
    ap.add_argument("--sparqlce", default=None, help="SparqlCE checkpoint path")
    ap.add_argument("--no-lexical", action="store_true",
                    help="skip TF-IDF/BM25 (e.g. when only adding a neural row)")
    ap.add_argument("--split", default=str(SPLIT),
                    help="path to the RQ1 split JSON (default: stratified split)")
    ap.add_argument("--stratify", action="store_true",
                    help="compute lexical bins on the fly (mean expanded-text cos "
                         "anchor->positives) and report per cos-bin, ignoring any "
                         "lexical_bin field in the split")
    args = ap.parse_args()

    split_path = Path(args.split)
    raw, corpus, items, item_bins = load_split_and_bins(split_path)
    print(f"split: {split_path.name}  anchors={len(items)}  corpus={len(corpus)}")

    # Wikidata's opaque P/Q ids starve every TEXT baseline of lexical signal;
    # expand them to English labels so text baselines read the same semantic
    # content SparqlCE's graph builder resolves internally. No-op on DBpedia
    # (no P/Q ids to match). SparqlCE keeps the raw `corpus` -- the expanded
    # text is not valid SPARQL syntax, only a plain-text stand-in.
    #
    # Loaded by file path, not `from sparqlce.sparql_utils import ...`: the
    # package __init__ pulls in torch, and the lexical rows are supposed to run
    # without it (see module docstring). sparql_utils itself is torch-free.
    expand_wikidata_labels = _load_label_expander()
    text_corpus = [expand_wikidata_labels(q) for q in corpus]

    # expand_wikidata_labels is a silent no-op on ids it cannot resolve, so a
    # missing -- or half-built, which build_wikidata_labels.py checkpoints to
    # disk on interrupt -- label map still "succeeds": the text baselines would
    # read raw P/Q ids while SparqlCE resolves labels internally, exactly the
    # preprocessing asymmetry the expansion exists to remove. Report coverage
    # and refuse to score an unfair comparison.
    if _is_wikidata(corpus):
        n_ids, n_resolved = _label_coverage(corpus, expand_wikidata_labels)
        pct = 100.0 * n_resolved / n_ids if n_ids else 0.0
        print(f"wikidata label coverage: {n_resolved}/{n_ids} distinct ids ({pct:.1f}%)")
        if pct < 99.0:
            sys.exit(
                f"!!! Only {pct:.1f}% of this split's P/Q ids resolve to a label.\n"
                "    datasets/wikidata_labels.json is missing or incomplete, so the\n"
                "    text baselines below would be scored on ids SparqlCE reads as\n"
                "    labels -- an unfair comparison. Build/finish the map with\n"
                "        python scripts/build_wikidata_labels.py\n"
                "    (it resumes from a partial map). To score raw ids for ALL\n"
                "    models on purpose, set SPARQLCE_NO_WIKIDATA_LABELS=1 -- that\n"
                "    switch applies to the graph builder too, keeping it fair."
            )

    if args.stratify:
        item_bins = stratify_by_lexical(corpus, text_corpus, items)
    from collections import Counter
    print("bins:", dict(Counter(item_bins)))

    if not args.no_lexical:
        run_tfidf(corpus, text_corpus, items, item_bins)
        run_bm25(corpus, text_corpus, items, item_bins)
    for model_id in args.st:
        run_st(model_id, corpus, text_corpus, items, item_bins)
    for model_id in getattr(args, "code_llm"):
        run_code_llm(model_id, corpus, text_corpus, items, item_bins)
    if args.sparqlce:
        run_sparqlce(args.sparqlce, corpus, items, item_bins)


if __name__ == "__main__":
    main()
