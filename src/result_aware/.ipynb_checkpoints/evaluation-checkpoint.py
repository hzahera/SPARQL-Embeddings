from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .data import QueryRecord, unique_queries
from .metrics import (
    hits_at_k,
    is_relevant,
    macro_mean,
    mrr_at_k,
    ndcg_at_k,
    precision_recall_f1_at_k,
    spearman_alignment,
)
from .model import HFEmbeddingModel
from .utils import normalize_query


@dataclass
class EncodedSplit:
    embeddings: dict[str, torch.Tensor]
    surface_forms: dict[str, str]

    def get(self, query: str) -> torch.Tensor:
        return self.embeddings[normalize_query(query)]


def encode_split(
    model: HFEmbeddingModel,
    records: list[QueryRecord],
    batch_size: int = 64,
    show_progress: bool = True,
) -> EncodedSplit:
    texts = unique_queries(records)
    vectors = model.encode(texts, batch_size=batch_size, show_progress=show_progress)
    embeddings: dict[str, torch.Tensor] = {}
    surfaces: dict[str, str] = {}
    for text, vec in zip(texts, vectors):
        key = normalize_query(text)
        embeddings[key] = vec.float()
        surfaces[key] = text
    return EncodedSplit(embeddings=embeddings, surface_forms=surfaces)


def rank_record(rec: QueryRecord, encoded: EncodedSplit) -> list[dict[str, Any]]:
    """Rank only the candidates explicitly attached to one anchor.

    This local ranking is retained for the semantic-cache simulation, whose
    behaviour intentionally remains unchanged. Retrieval evaluation uses
    :func:`rank_record_global` instead.
    """
    anchor = encoded.get(rec.anchor_query)
    ranked: list[dict[str, Any]] = []
    for cand in rec.candidates:
        score = float(torch.dot(anchor, encoded.get(cand.query)).item())
        ranked.append(
            {
                "query": cand.query,
                "cosine": score,
                "answer_jaccard": float(cand.answer_jaccard),
            }
        )
    ranked.sort(key=lambda x: x["cosine"], reverse=True)
    return ranked


def _global_relevance_map(rec: QueryRecord) -> dict[str, float]:
    """Known relevance judgments for one anchor, keyed by normalized query.

    Queries in the global corpus that were not annotated for this anchor are
    treated as non-relevant (0.0), matching standard pooled retrieval
    evaluation. Explicitly stored candidate Jaccard scores are preserved so
    graded nDCG still uses execution-derived relevance where it is known.
    """
    rel: dict[str, float] = {}
    for cand in rec.candidates:
        key = normalize_query(cand.query)
        # If a query appears more than once, keep the strongest known judgment.
        rel[key] = max(rel.get(key, 0.0), float(cand.answer_jaccard))
    return rel


def rank_record_global(rec: QueryRecord, encoded: EncodedSplit) -> list[dict[str, Any]]:
    """Rank the entire deduplicated split corpus for one anchor.

    The anchor itself and any per-anchor ``excluded_queries`` are physically
    removed from the ranking. All other unique queries encoded for the split
    are eligible retrieval candidates.
    """
    anchor_key = normalize_query(rec.anchor_query)
    anchor = encoded.embeddings[anchor_key]
    relevance = _global_relevance_map(rec)
    excluded = {
        normalize_query(q)
        for q in getattr(rec, "excluded_queries", [])
    }

    ranked: list[dict[str, Any]] = []
    for key, cand_emb in encoded.embeddings.items():
        if key == anchor_key or key in excluded:
            continue
        score = float(torch.dot(anchor, cand_emb).item())
        ranked.append(
            {
                "query": encoded.surface_forms[key],
                "cosine": score,
                "answer_jaccard": float(relevance.get(key, 0.0)),
            }
        )
    ranked.sort(key=lambda x: x["cosine"], reverse=True)
    return ranked

def evaluate_records(
    model: HFEmbeddingModel,
    records: list[QueryRecord],
    eval_cfg: dict[str, Any],
    split_name: str,
    include_per_anchor: bool = False,
) -> dict[str, Any]:
    ks = [int(k) for k in eval_cfg.get("ks", [1, 3, 5, 10])]
    threshold = float(eval_cfg.get("relevance_threshold", 0.8))
    operator = str(eval_cfg.get("relevance_operator", "ge"))
    gain = str(eval_cfg.get("ndcg_gain", "linear"))
    batch_size = int(eval_cfg.get("encode_batch_size", 64))

    encoded = encode_split(model, records, batch_size=batch_size, show_progress=True)

    ranking_values: dict[str, list[float]] = {}
    binary_values: dict[str, list[float]] = {}
    for k in ks:
        for name in ("mrr", "hits", "ndcg"):
            ranking_values[f"{name}@{k}"] = []
        for name in ("precision", "recall", "f1"):
            binary_values[f"{name}@{k}"] = []

    all_cosines: list[float] = []
    all_jaccards: list[float] = []
    per_anchor: list[dict[str, Any]] = []

    for rec in tqdm(records, desc=f"Evaluating {split_name}"):
        ranked = rank_record_global(rec, encoded)
        rels = [float(x["answer_jaccard"]) for x in ranked]
        binary = [is_relevant(x, threshold, operator) for x in rels]
        for k in ks:
            ranking_values[f"mrr@{k}"].append(mrr_at_k(binary, k))
            ranking_values[f"hits@{k}"].append(hits_at_k(binary, k))
            ranking_values[f"ndcg@{k}"].append(ndcg_at_k(rels, k, gain=gain))
            p, r, f1 = precision_recall_f1_at_k(binary, k)
            binary_values[f"precision@{k}"].append(p)
            binary_values[f"recall@{k}"].append(r)
            binary_values[f"f1@{k}"].append(f1)

        # Alignment remains defined only on explicitly annotated anchor-candidate
        # pairs. Global retrieval introduces many unannotated corpus queries, and
        # treating those as J=0 here would change the alignment experiment.
        local_ranked = rank_record(rec, encoded)
        all_cosines.extend(float(x["cosine"]) for x in local_ranked)
        all_jaccards.extend(float(x["answer_jaccard"]) for x in local_ranked)
        if include_per_anchor:
            per_anchor.append(
                {
                    "anchor_query": rec.anchor_query,
                    "ranked_candidates": ranked,
                }
            )

    result: dict[str, Any] = {
        "split": split_name,
        "anchors": len(records),
        "unique_encoded_queries": len(encoded.embeddings),
        "alignment": {
            "spearman_rho": spearman_alignment(all_cosines, all_jaccards),
            "pair_count": len(all_cosines),
        },
        "ranking": {k: macro_mean(v) for k, v in ranking_values.items()},
        "binary": {k: macro_mean(v) for k, v in binary_values.items()},
        "evaluation_definition": {
            "ks": ks,
            "binary_relevance": f"J {operator} {threshold}",
            "relevance_threshold": threshold,
            "relevance_operator": operator,
            "ndcg_gain": gain,
            "averaging": "macro over anchors",
            "candidate_universe": "all unique queries in the held-out split, excluding the anchor and per-anchor unjudged/excluded queries",
            "unannotated_global_queries": "treated as non-relevant (J=0.0)",
        },
    }
    if include_per_anchor:
        result["per_anchor"] = per_anchor
    return result


def evaluate_from_encoded(
    records: list[QueryRecord],
    encoded: EncodedSplit,
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Same metrics as evaluate_records, reusing a precomputed embedding index."""
    ks = [int(k) for k in eval_cfg.get("ks", [1, 3, 5])]
    threshold = float(eval_cfg.get("relevance_threshold", 0.8))
    operator = str(eval_cfg.get("relevance_operator", "ge"))
    gain = str(eval_cfg.get("ndcg_gain", "linear"))
    ranking_values = {f"{m}@{k}": [] for k in ks for m in ("mrr", "hits", "ndcg")}
    binary_values = {f"{m}@{k}": [] for k in ks for m in ("precision", "recall", "f1")}
    all_cosines: list[float] = []
    all_jaccards: list[float] = []

    for rec in records:
        ranked = rank_record_global(rec, encoded)
        rels = [float(x["answer_jaccard"]) for x in ranked]
        binary = [is_relevant(x, threshold, operator) for x in rels]
        for k in ks:
            ranking_values[f"mrr@{k}"].append(mrr_at_k(binary, k))
            ranking_values[f"hits@{k}"].append(hits_at_k(binary, k))
            ranking_values[f"ndcg@{k}"].append(ndcg_at_k(rels, k, gain=gain))
            p, r, f1 = precision_recall_f1_at_k(binary, k)
            binary_values[f"precision@{k}"].append(p)
            binary_values[f"recall@{k}"].append(r)
            binary_values[f"f1@{k}"].append(f1)
        local_ranked = rank_record(rec, encoded)
        all_cosines.extend(float(x["cosine"]) for x in local_ranked)
        all_jaccards.extend(float(x["answer_jaccard"]) for x in local_ranked)

    return {
        "alignment": {
            "spearman_rho": spearman_alignment(all_cosines, all_jaccards),
            "pair_count": len(all_cosines),
        },
        "ranking": {k: macro_mean(v) for k, v in ranking_values.items()},
        "binary": {k: macro_mean(v) for k, v in binary_values.items()},
    }
