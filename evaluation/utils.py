import json
import os
import sys
from typing import Any, Dict, List, Set, Tuple
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
_UTILS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _UTILS_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from src.embedder import SPARQLEmbedder

# ----------------------------
# Metrics
# ----------------------------

def avg_prf_at_k(sorted_pools, items, k):
    p_sum, r_sum, f1_sum, total = 0.0, 0.0, 0.0, 0

    for sorted_pool, (_, _, positives) in zip(sorted_pools, items):
        if not positives:
            continue

        pos_set = set(positives)
        overlap = len(set(sorted_pool[:k]) & pos_set)

        p = overlap / k if k > 0 else 0.0
        r = overlap / len(pos_set) if pos_set else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        p_sum += p
        r_sum += r
        f1_sum += f1
        total += 1

    if total == 0:
        return 0.0, 0.0, 0.0

    return p_sum / total, r_sum / total, f1_sum / total


def avg_mrr_at_k(sorted_pools, items, k):
    rr_sum, total = 0.0, 0

    for sorted_pool, (_, _, positives) in zip(sorted_pools, items):
        if not positives:
            continue

        pos_set = set(positives)
        rr = 0.0

        for rank, q in enumerate(sorted_pool[:k], start=1):
            if q in pos_set:
                rr = 1.0 / rank
                break

        rr_sum += rr
        total += 1

    return rr_sum / max(total, 1)


def avg_hits_at_k(sorted_pools, items, k):
    hits_sum, total = 0.0, 0

    for sorted_pool, (_, _, positives) in zip(sorted_pools, items):
        if not positives:
            continue

        pos_set = set(positives)
        hits_sum += int(any(q in pos_set for q in sorted_pool[:k]))
        total += 1

    return hits_sum / max(total, 1)


def dcg(relevances: List[float]) -> float:
    return sum(rel / np.log2(rank + 2) for rank, rel in enumerate(relevances))


def avg_ndcg_at_k(sorted_pools, k: int, relevance_scores_per_item: List[Dict[str, float]]) -> float:
    total, count = 0.0, 0

    for sorted_pool, relevance_map in zip(sorted_pools, relevance_scores_per_item):
        topk = [q for q in sorted_pool[:k] if q in relevance_map]
        if not topk:
            continue

        relevances = [relevance_map[q] for q in topk]
        dcg_value = dcg(relevances)

        ideal_relevances = sorted(relevance_map.values(), reverse=True)[:k]
        ideal_dcg = dcg(ideal_relevances)

        ndcg = dcg_value / ideal_dcg if ideal_dcg > 0 else 0.0
        total += ndcg
        count += 1

    return total / max(count, 1)


# ----------------------------
# DBpedia/Wikidata eval helpers
# ----------------------------

EvalItem = Tuple[str, List[str], Set[str]]


def get_query_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, dict):
        return str(x.get("query") or x.get("sparql") or "").strip()
    return str(x).strip()


def get_anchor_query(item: Dict[str, Any]) -> str:
    return get_query_text(item.get("anchor_query") or item.get("base_query"))


def iter_labeled_queries(item: Dict[str, Any], positives_only: bool) -> List[str]:
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


def build_sorted_pools_global(emb_map: Dict[str, np.ndarray], items: List[EvalItem], corpus_queries: List[str]) -> List[List[str]]:
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


def build_sorted_pools_tfidf(X, q_to_row: Dict[str, int], items: List[EvalItem], corpus_queries: List[str]) -> List[List[str]]:
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


def avg_prf_at_k_annotated(sorted_pools: List[List[str]], items: List[EvalItem], k: int) -> Tuple[float, float, float]:
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


def avg_mrr_at_k_annotated(sorted_pools: List[List[str]], items: List[EvalItem], k: int) -> float:
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


def avg_hits_at_k_annotated(sorted_pools: List[List[str]], items: List[EvalItem], k: int) -> float:
    hits = 0
    total = 0
    for sorted_pool, (_, _, positives) in zip(sorted_pools, items):
        if not positives:
            continue
        total += 1
        if any(q in positives for q in sorted_pool[:k]):
            hits += 1
    return hits / max(total, 1)


def avg_ndcg_binary(sorted_pools: List[List[str]], rel_maps: List[Dict[str, float]], k: int) -> float:
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


def avg_mean_rel_at_k(sorted_pools: List[List[str]], rel_maps: List[Dict[str, float]], k: int) -> float:
    total, count = 0.0, 0
    for sorted_pool, rel_map in zip(sorted_pools, rel_maps):
        topk = [q for q in sorted_pool[:k] if q in rel_map]
        if not topk:
            continue
        total += float(np.mean([rel_map[q] for q in topk]))
        count += 1
    return total / max(count, 1)


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


def evaluate_tfidf(
    eval_items: List[EvalItem],
    global_corpus: List[str],
    relevance_scores_per_item: List[Dict[str, float]],
    topks: List[int],
) -> None:
    tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(global_corpus)
    X_tfidf = sk_normalize(X_tfidf, norm="l2", axis=1)
    q_to_row = {q: i for i, q in enumerate(global_corpus)}
    tfidf_sorted = build_sorted_pools_tfidf(X_tfidf, q_to_row, eval_items, global_corpus)
    report("Baseline: TF-IDF (global corpus)", tfidf_sorted, eval_items, relevance_scores_per_item, topks)


def evaluate_sentence_transformer(
    model_name: str,
    display_name: str,
    eval_items: List[EvalItem],
    global_corpus: List[str],
    relevance_scores_per_item: List[Dict[str, float]],
    topks: List[int],
    batch_size: int = 32,
    transform=None,
) -> None:
    model = SentenceTransformer(model_name)
    emb_map = build_emb_map_sentence_transformer(model, global_corpus, batch_size=batch_size, transform=transform)
    sorted_pools = build_sorted_pools_global(emb_map, eval_items, global_corpus)
    report(display_name, sorted_pools, eval_items, relevance_scores_per_item, topks)


def evaluate_sparqlce_checkpoint(
    checkpoint_path: str,
    eval_items: List[EvalItem],
    global_corpus: List[str],
    relevance_scores_per_item: List[Dict[str, float]],
    topks: List[int],
    batch_size: int = 32,
) -> None:
    if not os.path.isfile(checkpoint_path):
        print(f"Skip SPARQLCE: checkpoint not found at {checkpoint_path}")
        return
    model = SPARQLEmbedder(checkpoint_path)
    emb_map = build_emb_map_sparqlce(model, global_corpus, batch_size=batch_size)
    sorted_pools = build_sorted_pools_global(emb_map, eval_items, global_corpus)
    report("SPARQLCE", sorted_pools, eval_items, relevance_scores_per_item, topks)