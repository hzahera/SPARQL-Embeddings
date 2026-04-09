import json
import os
import re
from typing import Dict, List, Set

import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON as SPARQL_JSON

CACHE_FILE = "sparql_cache.json"

ENDPOINTS = {
    "wikidata": "https://wikidata.data.dice-research.org/sparql",
    "dbpedia": "https://dbpedia.data.dice-research.org/sparql"
}


# ----------------------------
# SPARQL Cache Utilities
# ----------------------------

def save_answer_cache(cache_data: Dict[str, Set[str]], filepath: str = CACHE_FILE) -> None:
    """Save answer cache to a JSON file."""
    serializable_cache = {key: list(answers) for key, answers in cache_data.items()}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_cache, f, indent=4)


def load_answer_cache(filepath: str = CACHE_FILE) -> Dict[str, Set[str]]:
    """Load answer cache from a JSON file."""
    if not os.path.exists(filepath):
        return {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        return {key: set(answers) for key, answers in loaded_data.items()}
    except json.JSONDecodeError:
        return {}


answer_cache: Dict[str, Set[str]] = load_answer_cache()


def normalize_query(q: str) -> str:
    return "\n".join(line.strip() for line in q.strip().splitlines())


def get_answers(query: str, endpoint_name: str = "dbpedia", limit: int = 200) -> Set[str]:
    """
    Execute query on selected endpoint and cache the answers.

    Args:
        query: SPARQL query string
        endpoint_name: 'wikidata' or 'dbpedia'
        limit: maximum number of answers

    Returns:
        Set of answer strings
    """
    if endpoint_name not in ENDPOINTS:
        raise ValueError(
            f"Unknown endpoint '{endpoint_name}'. Choose from {list(ENDPOINTS.keys())}"
        )

    endpoint_url = ENDPOINTS[endpoint_name]
    q_norm = normalize_query(query)
    cache_key = f"{endpoint_name}::{q_norm}"

    if cache_key in answer_cache:
        return answer_cache[cache_key]

    endpoint = SPARQLWrapper(endpoint_url)
    endpoint.setReturnFormat(SPARQL_JSON)
    endpoint.setQuery(query + f"\nLIMIT {limit}")

    try:
        res = endpoint.query().convert()
        bindings = res.get("results", {}).get("bindings", [])
    except Exception:
        bindings = []
        res = {"head": {"vars": []}}

    answers: Set[str] = set()

    if bindings:
        vars_ = res.get("head", {}).get("vars", [])
        for b in bindings:
            vals = []
            for v in vars_:
                if v in b:
                    vals.append(b[v]["value"])
            if vals:
                answers.add("|".join(vals))

    answer_cache[cache_key] = answers
    return answers


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


def avg_jaccard_at_k(sorted_pools, k: int, relevance_scores_per_item: List[Dict[str, float]]) -> float:
    """
    Average relevance score at k.
    relevance_scores_per_item: list of {query_str -> score}
    """
    total, count = 0.0, 0

    for sorted_pool, relevance_map in zip(sorted_pools, relevance_scores_per_item):
        topk = [q for q in sorted_pool[:k] if q in relevance_map]
        if not topk:
            continue

        total += np.mean([relevance_map[q] for q in topk])
        count += 1

    return total / max(count, 1)


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