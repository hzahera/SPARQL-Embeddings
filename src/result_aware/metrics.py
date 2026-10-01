from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr


def is_relevant(jaccard: float, threshold: float = 0.8, operator: str = "ge") -> bool:
    if operator == "ge":
        return jaccard >= threshold
    if operator == "gt":
        return jaccard > threshold
    raise ValueError(f"Unknown relevance operator {operator!r}")


def dcg(relevances: Iterable[float], gain: str = "linear") -> float:
    vals = list(float(x) for x in relevances)
    total = 0.0
    for rank, rel in enumerate(vals, start=1):
        if gain == "linear":
            g = rel
        elif gain == "exp2":
            g = 2.0 ** rel - 1.0
        else:
            raise ValueError(f"Unknown gain {gain!r}")
        total += g / math.log2(rank + 1.0)
    return total


def ndcg_at_k(relevances: list[float], k: int, gain: str = "linear") -> float:
    actual = relevances[:k]
    ideal = sorted(relevances, reverse=True)[:k]
    denom = dcg(ideal, gain=gain)
    if denom <= 0:
        return 0.0
    return dcg(actual, gain=gain) / denom


def mrr_at_k(binary_relevance: list[bool], k: int) -> float:
    for rank, relevant in enumerate(binary_relevance[:k], start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def hits_at_k(binary_relevance: list[bool], k: int) -> float:
    return float(any(binary_relevance[:k]))


def precision_recall_f1_at_k(binary_relevance: list[bool], k: int) -> tuple[float, float, float]:
    top = binary_relevance[:k]
    tp = sum(top)
    # Standard P@k denominator. In this benchmark every retained anchor has >=20 candidates.
    precision = tp / float(k)
    total_relevant = sum(binary_relevance)
    recall = tp / float(total_relevant) if total_relevant else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def spearman_alignment(cosines: list[float], jaccards: list[float]) -> float:
    if len(cosines) != len(jaccards):
        raise ValueError("cosines and jaccards must have equal length")
    if len(cosines) < 2:
        return float("nan")
    result = spearmanr(cosines, jaccards)
    rho = float(result.statistic)
    return rho


def macro_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)))
