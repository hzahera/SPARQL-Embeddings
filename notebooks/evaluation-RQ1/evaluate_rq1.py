#!/usr/bin/env python3
"""RQ1 Evaluation Script — Uses SPARQLEmbedder directly (patches already built into graph_builder)."""

import json
import os
import sys
from typing import Any, Dict, List, Set, Tuple
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from rank_bm25 import BM25Okapi

# ---- Project path setup (Notebook & Script Safe) ----
try:
    # Works if running as a standard .py script
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Falls back to Current Working Directory if running in a Jupyter Notebook
    _PROJECT_ROOT = Path.cwd().parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# The graph_builder already has:
#   - canonicalize_variables() built into build_graph()
#   - is_noop_filter_text() to skip FILTER(true)
# So we can use SPARQLEmbedder directly — no patches needed.
from sparqlce.embedder import SPARQLEmbedder
# ----------------------------
# Metrics
# ----------------------------

def dcg(relevances: List[float]) -> float:
    return sum(rel / np.log2(rank + 2) for rank, rel in enumerate(relevances))

# ----------------------------
# Data loading helpers
# ----------------------------

EvalItem = Tuple[str, List[str], Set[str]]

def get_query_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, dict):
        return str(x.get("query") or x.get("sparql") or "").strip()
    return str(x).strip()

def get_anchor_query(item: Dict[str, Any]) -> str:
    return get_query_text(item.get("anchor_query") or item.get("query") or item.get("base_query"))

def iter_labeled_queries(item: Dict[str, Any], positives_only: bool = False) -> List[str]:
    out: List[str] = []
    for p in item.get("positive_queries") or []:
        q = get_query_text(p)
        if q:
            out.append(q)
    if not positives_only:
        for n in item.get("negative_queries") or []:
            q = get_query_text(n)
            if q:
                out.append(q)
    return out

def load_triplets(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_global_corpus(raw_data: List[Dict[str, Any]]) -> Tuple[List[str], Set[str]]:
    unique_queries: List[str] = []
    seen_q: Set[str] = set()

    def add_unique(q: str) -> None:
        if q and q not in seen_q:
            seen_q.add(q)
            unique_queries.append(q)

    for item in raw_data:
        anchor_q = get_anchor_query(item)
        if not anchor_q:
            continue
        add_unique(anchor_q)
        for q in iter_labeled_queries(item, positives_only=False):
            add_unique(q)

    return unique_queries, seen_q

def build_eval_items(raw_data: List[Dict[str, Any]], global_corpus: List[str], seen_q: Set[str]) -> List[EvalItem]:
    eval_items: List[EvalItem] = []
    for item in raw_data:
        anchor_q = get_anchor_query(item)
        if not anchor_q or anchor_q not in seen_q:
            continue

        pool = [q for q in global_corpus if q != anchor_q]
        positives: Set[str] = set()
        for p in item.get("positive_queries") or []:
            q = get_query_text(p)
            if q and q in seen_q and q != anchor_q:
                positives.add(q)

        if positives:
            eval_items.append((anchor_q, pool, positives))
    return eval_items

def build_binary_relevance_maps(eval_items: List[EvalItem], global_corpus: List[str]) -> List[Dict[str, float]]:
    relevance_scores_per_item: List[Dict[str, float]] = []
    for _, _, positives in eval_items:
        scores = {cand: (1.0 if cand in positives else 0.0) for cand in global_corpus}
        relevance_scores_per_item.append(scores)
    return relevance_scores_per_item

# ----------------------------
# Embedding builders
# ----------------------------

def build_emb_map_sentence_transformer(
    model: SentenceTransformer,
    queries: List[str],
    batch_size: int = 32,
    transform=None,
    normalize_embeddings: bool = True,
) -> Dict[str, np.ndarray]:
    emb_map: Dict[str, np.ndarray] = {}
    for i in range(0, len(queries), batch_size):
        batch_qs = queries[i : i + batch_size]
        texts = [transform(q) for q in batch_qs] if transform else batch_qs
        batch_embs = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
        for q, e in zip(batch_qs, batch_embs):
            emb_map[q] = e
    return emb_map

def build_emb_map_sparqlce(
    model: SPARQLEmbedder,
    queries: List[str],
    batch_size: int = 32,
    normalize_embeddings: bool = True,
) -> Dict[str, np.ndarray]:
    """Encode queries using the original SPARQLEmbedder (handles canonicalization + FILTER(true) skip)."""
    emb_map: Dict[str, np.ndarray] = {}
    for i in range(0, len(queries), batch_size):
        batch_qs = queries[i : i + batch_size]
        embs = model.encode(batch_qs)
        if not isinstance(embs, np.ndarray):
            embs = np.array(embs)
        if normalize_embeddings:
            embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        for q, e in zip(batch_qs, embs):
            emb_map[q] = e
    return emb_map

# ----------------------------
# Sorting / retrieval
# ----------------------------

def build_sorted_pools_global(
    emb_map: Dict[str, np.ndarray],
    items: List[EvalItem],
    corpus_queries: List[str]
) -> List[List[str]]:
    corpus_embs = np.vstack([emb_map[q] for q in corpus_queries])
    anchor_to_idx = {q: i for i, q in enumerate(corpus_queries)}
    sorted_pools: List[List[str]] = []

    for anchor_q, _, _ in items:
        a_emb = emb_map[anchor_q]
        scores = corpus_embs @ a_emb
        if anchor_q in anchor_to_idx:
            scores = scores.copy()
            scores[anchor_to_idx[anchor_q]] = np.finfo(scores.dtype).min
        order = np.argsort(-scores)
        sorted_pools.append([corpus_queries[i] for i in order])
    return sorted_pools

def build_sorted_pools_bm25(
    bm25: BM25Okapi,
    q_to_row: Dict[str, int],
    items: List[EvalItem],
    corpus_queries: List[str],
    transform=None,
) -> List[List[str]]:
    anchor_to_idx = q_to_row
    sorted_pools: List[List[str]] = []

    for anchor_q, _, _ in items:
        # Anchor tokens must match how the corpus was tokenized: if the corpus
        # was expanded (transform set), expand the anchor query too.
        tokens = (transform(anchor_q) if transform else anchor_q).lower().split()
        scores = np.asarray(bm25.get_scores(tokens))
        if anchor_q in anchor_to_idx:
            scores = scores.copy()
            scores[anchor_to_idx[anchor_q]] = np.finfo(scores.dtype).min
        order = np.argsort(-scores)
        sorted_pools.append([corpus_queries[i] for i in order])
    return sorted_pools

def build_sorted_pools_tfidf(
    X, q_to_row: Dict[str, int],
    items: List[EvalItem],
    corpus_queries: List[str]
) -> List[List[str]]:
    corpus_vecs = X
    anchor_to_idx = {q: i for i, q in enumerate(corpus_queries)}
    sorted_pools: List[List[str]] = []

    for anchor_q, _, _ in items:
        a_vec = X[q_to_row[anchor_q]]
        sims = corpus_vecs @ a_vec.T
        if hasattr(sims, "toarray"):
            sims = sims.toarray().ravel()
        else:
            sims = np.asarray(sims).ravel()
        if anchor_q in anchor_to_idx:
            sims = sims.copy()
            sims[anchor_to_idx[anchor_q]] = np.finfo(sims.dtype).min
        order = np.argsort(-sims)
        sorted_pools.append([corpus_queries[i] for i in order])
    return sorted_pools

# ----------------------------
# Metrics (annotated positives)
# ----------------------------

def avg_prf_at_k_annotated(sorted_pools, items, k):
    p_sum = r_sum = f1_sum = 0.0
    total = 0
    for sorted_pool, (_, _, positives) in zip(sorted_pools, items):
        if not positives:
            continue
        topk = sorted_pool[:k]
        inter = len(set(topk) & positives)
        p = inter / k if k > 0 else 0.0
        r = inter / len(positives) if positives else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        p_sum += p
        r_sum += r
        f1_sum += f1
        total += 1
    if total == 0:
        return 0.0, 0.0, 0.0
    return p_sum / total, r_sum / total, f1_sum / total

def avg_mrr_at_k_annotated(sorted_pools, items, k):
    rr_sum = 0.0
    total = 0
    for sorted_pool, (_, _, positives) in zip(sorted_pools, items):
        if not positives:
            continue
        total += 1
        rr = 0.0
        for rank, q in enumerate(sorted_pool[:k], start=1):
            if q in positives:
                rr = 1.0 / rank
                break
        rr_sum += rr
    return rr_sum / max(total, 1)

def avg_hits_at_k_annotated(sorted_pools, items, k):
    hits = 0
    total = 0
    for sorted_pool, (_, _, positives) in zip(sorted_pools, items):
        if not positives:
            continue
        total += 1
        if any(q in positives for q in sorted_pool[:k]):
            hits += 1
    return hits / max(total, 1)

def avg_ndcg_binary(sorted_pools, rel_maps, k):
    total, count = 0.0, 0
    for sorted_pool, rel_map in zip(sorted_pools, rel_maps):
        topk = [q for q in sorted_pool[:k] if q in rel_map]
        if not topk:
            continue
        rels = [rel_map[q] for q in topk]
        dcg_val = dcg(rels)
        ideal_rels = sorted(rel_map.values(), reverse=True)[:k]
        idcg = dcg(ideal_rels)
        total += (dcg_val / idcg) if idcg > 0 else 0.0
        count += 1
    return total / max(count, 1)

def avg_mean_rel_at_k(sorted_pools, rel_maps, k):
    total, count = 0.0, 0
    for sorted_pool, rel_map in zip(sorted_pools, rel_maps):
        topk = [q for q in sorted_pool[:k] if q in rel_map]
        if not topk:
            continue
        total += float(np.mean([rel_map[q] for q in topk]))
        count += 1
    return total / max(count, 1)

# ----------------------------
# Report
# ----------------------------

# When set to a list of cumulative TF-IDF-cosine thresholds (e.g. [0.3, 0.5,
# 0.8]), every report() call also emits a per-threshold breakdown of MRR
# restricted to positives with cos(anchor, positive) < threshold -- cumulative
# from 0, not a partition -- so we can show WHERE each method wins (lexical
# methods hold on high-similarity positives, collapse on surface-divergent
# ones; SparqlCE stays flat).
# Set from a notebook:  import evaluate_rq1; evaluate_rq1.STRATIFY_BINS = [0.3, 0.5, 0.8]
STRATIFY_BINS = None


def _lexical_cos_fn(eval_items):
    """Fit TF-IDF on the anchors+positives in eval_items; return cos(anchor, pos)."""
    texts = set()
    for a, _pool, pos in eval_items:
        texts.add(a)
        texts.update(pos)
    texts = list(texts)
    ridx = {q: i for i, q in enumerate(texts)}
    Xv = sk_normalize(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2)).fit_transform(texts),
        norm="l2", axis=1,
    )
    def cos(a, p):
        if a not in ridx or p not in ridx:
            return 0.0
        return float(Xv[ridx[a]].multiply(Xv[ridx[p]]).sum())
    return cos


def report_stratified(sorted_pools, eval_items, topks, thresholds):
    """Cumulative MRR by anchor-positive lexical (TF-IDF cosine) similarity.
    `thresholds` (e.g. [0.3, 0.5, 0.8]) each define a slice cos(anchor,
    positive) < threshold, cumulative from 0 -- NOT a partition, so the
    slices overlap/nest (every positive in the <0.3 slice is also in <0.5
    and <0.8). The printed `mean_cos` / `median_cos` columns report the
    actual TF-IDF cosine values observed within each slice, so the number
    that gets quoted is never just the threshold label alone."""
    cos = _lexical_cos_fn(eval_items)
    print("  -- stratified by anchor-positive lexical similarity (TF-IDF cosine, cumulative) --")
    print(f"    {'lexical bin':>12} {'mean_cos':>9} {'median_cos':>11} {'anchors':>8} " +
          " ".join(f"MRR@{k:<4}" for k in topks))
    for t in thresholds:
        sp_sub, it_sub, cos_vals = [], [], []
        for sp, (a, pool, pos) in zip(sorted_pools, eval_items):
            scored = {p: cos(a, p) for p in pos}
            inb = {p for p, c in scored.items() if c < t}
            if inb:
                sp_sub.append(sp)
                it_sub.append((a, pool, inb))
                cos_vals.extend(scored[p] for p in inb)
        label = f"<{t:.1f}"
        if not it_sub:
            print(f"    {label:>12} {'--':>9} {'--':>11} {0:>8}   (no anchors)")
            continue
        mean_cos = float(np.mean(cos_vals))
        median_cos = float(np.median(cos_vals))
        mrrs = [avg_mrr_at_k_annotated(sp_sub, it_sub, k) for k in topks]
        print(f"    {label:>12} {mean_cos:9.4f} {median_cos:11.4f} {len(it_sub):>8} " +
              " ".join(f"{m:8.4f}" for m in mrrs))


def report(
    name: str,
    sorted_pools: List[List[str]],
    eval_items: List[EvalItem],
    relevance_scores_per_item: List[Dict[str, float]],
    topks: List[int],
) -> None:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print("Binary relevance (annotated positives):")
    print(f"{'':5} {'P':>7} {'R':>7} {'F1':>7} {'MRR':>7} {'Hits':>7} {'meanRel':>9} {'nDCG':>7}")
    print("-" * 62)
    for k in topks:
        p, r, f1 = avg_prf_at_k_annotated(sorted_pools, eval_items, k)
        mrr = avg_mrr_at_k_annotated(sorted_pools, eval_items, k)
        hits = avg_hits_at_k_annotated(sorted_pools, eval_items, k)
        mr = avg_mean_rel_at_k(sorted_pools, relevance_scores_per_item, k)
        nd = avg_ndcg_binary(sorted_pools, relevance_scores_per_item, k)
        print(f"@{k:<4} {p:>7.4f} {r:>7.4f} {f1:>7.4f} {mrr:>7.4f} {hits:>7.4f} {mr:>9.4f} {nd:>7.4f}")
    if STRATIFY_BINS is not None:
        report_stratified(sorted_pools, eval_items, topks, STRATIFY_BINS)

# ----------------------------
# Evaluators
# ----------------------------

def evaluate_tfidf(eval_items, global_corpus, relevance_scores_per_item, topks, transform=None):
    # Fit on optionally label-expanded text, but keep q_to_row / lookups keyed
    # by the ORIGINAL query string so eval_items still resolve. The anchor's row
    # is its expanded vector, matching the corpus. (transform is a no-op unless
    # passed — DBpedia callers are unaffected.)
    corpus_text = [transform(q) for q in global_corpus] if transform else global_corpus
    tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(corpus_text)
    X_tfidf = sk_normalize(X_tfidf, norm="l2", axis=1)
    q_to_row = {q: i for i, q in enumerate(global_corpus)}
    tfidf_sorted = build_sorted_pools_tfidf(X_tfidf, q_to_row, eval_items, global_corpus)
    report("Baseline: TF-IDF", tfidf_sorted, eval_items, relevance_scores_per_item, topks)

def evaluate_bm25(eval_items, global_corpus, relevance_scores_per_item, topks, transform=None):
    corpus_text = [transform(q) for q in global_corpus] if transform else global_corpus
    tokenized_corpus = [q.lower().split() for q in corpus_text]
    bm25 = BM25Okapi(tokenized_corpus)
    q_to_row = {q: i for i, q in enumerate(global_corpus)}
    bm25_sorted = build_sorted_pools_bm25(bm25, q_to_row, eval_items, global_corpus, transform=transform)
    report("Baseline: BM25", bm25_sorted, eval_items, relevance_scores_per_item, topks)

def evaluate_sentence_transformer(model_name, display_name, eval_items, global_corpus, relevance_scores_per_item, topks, batch_size=32, transform=None):
    model = SentenceTransformer(model_name)
    emb_map = build_emb_map_sentence_transformer(model, global_corpus, batch_size=batch_size, transform=transform)
    sorted_pools = build_sorted_pools_global(emb_map, eval_items, global_corpus)
    report(display_name, sorted_pools, eval_items, relevance_scores_per_item, topks)

def evaluate_sparqlce_checkpoint(checkpoint_path, eval_items, global_corpus, relevance_scores_per_item, topks, batch_size=32):
    if not os.path.isfile(checkpoint_path):
        print(f"Skip SPARQLCE: checkpoint not found at {checkpoint_path}")
        return
    model = SPARQLEmbedder(checkpoint_path)
    emb_map = build_emb_map_sparqlce(model, global_corpus, batch_size=batch_size)
    sorted_pools = build_sorted_pools_global(emb_map, eval_items, global_corpus)
    report("SPARQLCE", sorted_pools, eval_items, relevance_scores_per_item, topks)


# ----------------------------
# Prior-literature SPARQL / code encoder baselines (zero-shot)
# ----------------------------
# Encoders used to represent SPARQL or code in prior work. Loaded as raw HF
# encoders; SentenceTransformer wraps each with MEAN pooling.
#
# NOTE: the "No sentence-transformers model found ... Creating a new one with
# MEAN pooling" notice these emit is the CORRECT behavior here — they are
# plain MLM encoders with no pooling head, and mean pooling is the standard
# way to obtain a sentence embedding from them. This is NOT the LoRA-adapter
# mis-pooling bug guarded against in scripts/eval_rq1_export.py (that concerned
# a fine-tuned adapter trained with last-token pooling).
SPARQL_ENCODER_BASELINES = [
    ("microsoft/codebert-base",      "CodeBERT (zero-shot)"),       # SparqlCE's own backbone, zero-shot
    ("microsoft/graphcodebert-base", "GraphCodeBERT (zero-shot)"),  # CodeBERT + data-flow graph
    ("razent/spbert-mlm-wso-base",   "SPBERT (zero-shot)"),         # BERT pretrained on SPARQL (alt ids: razent/spbert-mlm-zero)
]


def evaluate_sparql_encoder_baselines(eval_items, global_corpus, relevance_scores_per_item, topks, batch_size=32, transform=None):
    """Evaluate the prior-literature SPARQL/code encoder baselines (zero-shot,
    mean-pooled). Roster: SPARQL_ENCODER_BASELINES. Each model is wrapped in a
    try/except so one unavailable checkpoint doesn't abort the rest."""
    for model_id, display_name in SPARQL_ENCODER_BASELINES:
        try:
            evaluate_sentence_transformer(
                model_id, display_name, eval_items, global_corpus,
                relevance_scores_per_item, topks, batch_size=batch_size, transform=transform)
        except Exception as e:
            print(f"Skip {display_name} ({model_id}): {e}")


# ----------------------------
# Generative code-LLM baseline (decoder model used off-label as an embedder)
# ----------------------------
# Qwen2.5-Coder is a *generative* code LLM, not a purpose-built embedder. We
# embed it the standard way a decoder LM is turned into an embedder — mean/last
# hidden state of the final non-padding token (last-token pooling), L2-norm.
# Scale note: Qwen2.5-Coder-0.5B is chosen to roughly match the ~0.6B dense
# Qwen3-Embedding it replaces, keeping backbone scale comparable. This is a
# code-specialized LLM reference point; it is not trained for retrieval, so
# treat its zero-shot embedding quality accordingly.
def _causal_last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    idx = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[idx, seq_lens]


def build_emb_map_code_llm(model_id, queries, batch_size=16, max_length=512, transform=None):
    """Encode queries with a decoder code-LLM via last-token pooling.

    If ``transform`` is given, each query is transformed (e.g. Wikidata label
    expansion) before encoding, but emb_map stays keyed by the ORIGINAL query.
    """
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    emb_map = {}
    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            batch = list(queries[i:i + batch_size])
            texts = [transform(q) for q in batch] if transform else batch
            enc = tokenizer(texts, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            pooled = _causal_last_token_pool(out.last_hidden_state, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            for q, e in zip(batch, pooled.cpu().numpy()):
                emb_map[q] = e
    return emb_map


def evaluate_code_llm(model_id, display_name, eval_items, global_corpus, relevance_scores_per_item, topks, batch_size=16, transform=None):
    """Zero-shot decoder code-LLM (e.g. Qwen2.5-Coder) as an embedder via
    last-token pooling. Off-label use of a generative model; see note above."""
    try:
        emb_map = build_emb_map_code_llm(model_id, global_corpus, batch_size=batch_size, transform=transform)
        sorted_pools = build_sorted_pools_global(emb_map, eval_items, global_corpus)
        report(display_name, sorted_pools, eval_items, relevance_scores_per_item, topks)
    except Exception as e:
        print(f"Skip {display_name} ({model_id}): {e}")