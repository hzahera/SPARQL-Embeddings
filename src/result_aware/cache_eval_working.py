from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import QueryRecord
from .metrics import is_relevant
from .model import HFEmbeddingModel


# ============================================================
# Data structures
# ============================================================


@dataclass(frozen=True)
class OnlineCacheContext:
    """
    Precomputed information for one validation/test split.

    queries:
        Global unique query universe for the split:
        anchors + all attached candidate queries.

    similarities:
        Pairwise cosine-similarity matrix between all unique queries.

    jaccard:
        Stored answer-set Jaccard values for labelled query pairs.

    known_pairs:
        Query pairs for which the dataset explicitly contains a Jaccard label.

    query_to_idx:
        Mapping from query text to row/column in the similarity matrix.
    """

    queries: list[str]
    similarities: np.ndarray
    jaccard: dict[tuple[str, str], float]
    known_pairs: set[tuple[str, str]]
    query_to_idx: dict[str, int]


# ============================================================
# Query helpers
# ============================================================


def _candidate_query(candidate: Any) -> str:
    """
    Obtain the SPARQL string from a candidate object.

    QueryRecord candidates in the current project expose a query field,
    but several common alternatives are supported for robustness.
    """
    for attr in (
        "query",
        "candidate_query",
        "sparql",
        "text",
    ):
        value = getattr(candidate, attr, None)

        if isinstance(value, str):
            return value

    if isinstance(candidate, dict):
        for key in (
            "query",
            "candidate_query",
            "sparql",
            "text",
        ):
            value = candidate.get(key)

            if isinstance(value, str):
                return value

    raise AttributeError(
        "Could not determine candidate query text. "
        "Expected query/candidate_query/sparql/text."
    )


def _global_unique_queries(
    records: list[QueryRecord],
) -> list[str]:
    """
    Construct the same kind of global unique query universe used by the
    retrieval evaluation:

        unique anchors
        + unique positive candidates
        + unique negative candidates

    Each textual query appears exactly once in the online workload.
    """
    seen: set[str] = set()
    queries: list[str] = []

    for rec in records:
        anchor = rec.anchor_query

        if anchor not in seen:
            seen.add(anchor)
            queries.append(anchor)

        for candidate in rec.candidates:
            query = _candidate_query(candidate)

            if query not in seen:
                seen.add(query)
                queries.append(query)

    return queries


# ============================================================
# Ground-truth construction
# ============================================================


def _build_pairwise_labels(
    records: list[QueryRecord],
) -> tuple[
    dict[tuple[str, str], float],
    set[tuple[str, str]],
]:
    """
    Build the available pairwise result-equivalence ground truth.

    The dataset explicitly stores answer-set Jaccard between an anchor and
    its attached candidates.

    Jaccard is symmetric, therefore both:

        (anchor, candidate)
        (candidate, anchor)

    are stored.

    IMPORTANT:
        These labels are NEVER used by the runtime semantic-cache decision.
        They are used only after a decision to evaluate whether reuse was
        actually correct, and by the non-deployable oracle baseline.
    """
    jaccard: dict[tuple[str, str], float] = {}
    known_pairs: set[tuple[str, str]] = set()

    for rec in records:
        anchor = rec.anchor_query

        # Exact query identity is trivially result-equivalent.
        jaccard[(anchor, anchor)] = 1.0
        known_pairs.add((anchor, anchor))

        for candidate in rec.candidates:
            other = _candidate_query(candidate)
            value = float(candidate.answer_jaccard)

            for pair in (
                (anchor, other),
                (other, anchor),
            ):
                if pair in jaccard:
                    # Defensive handling in case the same labelled pair
                    # appears more than once.
                    jaccard[pair] = max(
                        jaccard[pair],
                        value,
                    )
                else:
                    jaccard[pair] = value

                known_pairs.add(pair)

            jaccard[(other, other)] = 1.0
            known_pairs.add((other, other))

    return jaccard, known_pairs


# ============================================================
# Embedding helpers
# ============================================================


def _encode_queries(
    model: HFEmbeddingModel,
    queries: list[str],
    batch_size: int,
) -> np.ndarray:
    """
    Encode every unique workload query once.
    """
    if not queries:
        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    encode_fn = getattr(
        model,
        "encode",
        None,
    )

    if not callable(encode_fn):
        raise AttributeError(
            "HFEmbeddingModel must expose encode()."
        )

    try:
        encoded = encode_fn(
            queries,
            batch_size=batch_size,
            show_progress=True,
        )
    except TypeError:
        encoded = encode_fn(
            queries,
            batch_size=batch_size,
        )

    if hasattr(encoded, "detach"):
        encoded = (
            encoded
            .detach()
            .cpu()
            .numpy()
        )

    embeddings = np.asarray(
        encoded,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Expected encode() to return a 2-D embedding matrix; "
            f"received shape={embeddings.shape}."
        )

    # Explicit normalization means cosine similarity can be computed
    # efficiently using a dot product.
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-12,
    )

    embeddings = embeddings / norms

    return embeddings


def _build_online_context(
    model: HFEmbeddingModel,
    records: list[QueryRecord],
    batch_size: int,
) -> OnlineCacheContext:
    queries = _global_unique_queries(records)

    query_to_idx = {
        query: idx
        for idx, query in enumerate(queries)
    }

    embeddings = _encode_queries(
        model=model,
        queries=queries,
        batch_size=batch_size,
    )

    if len(queries) > 0:
        similarities = (
            embeddings
            @ embeddings.T
        )

        similarities = np.clip(
            similarities,
            -1.0,
            1.0,
        )
    else:
        similarities = np.empty(
            (0, 0),
            dtype=np.float32,
        )

    jaccard, known_pairs = (
        _build_pairwise_labels(records)
    )

    return OnlineCacheContext(
        queries=queries,
        similarities=similarities,
        jaccard=jaccard,
        known_pairs=known_pairs,
        query_to_idx=query_to_idx,
    )


# ============================================================
# Pair labels
# ============================================================


def _pair_label(
    context: OnlineCacheContext,
    query_idx: int,
    cached_idx: int,
    relevance_threshold: float,
    relevance_operator: str,
    negative_jaccard_max: float = 0.1,
) -> str:
    """
    Classify the selected query pair for OFFLINE evaluation only.

    Returns one of:

        "positive"      J >= positive threshold (normally 0.8)
        "intermediate"  negative_max < J < positive threshold
        "negative"      J <= negative threshold (normally 0.1)
        "unknown"       no stored Jaccard label exists for the pair

    The semantic-cache runtime decision NEVER uses this label. Runtime reuse
    is decided only from cosine similarity and the fixed cosine threshold.
    """
    query = context.queries[query_idx]
    cached_query = context.queries[cached_idx]

    if query == cached_query:
        return "positive"

    pair = (query, cached_query)

    if pair not in context.known_pairs:
        return "unknown"

    value = float(context.jaccard[pair])

    # The project's current evaluation uses >= 0.8. Keep the operator
    # parameter for compatibility, but use the shared is_relevant helper.
    if is_relevant(
        value,
        relevance_threshold,
        relevance_operator,
    ):
        return "positive"

    if value <= float(negative_jaccard_max):
        return "negative"

    return "intermediate"


def _pair_jaccard(
    context: OnlineCacheContext,
    query_idx: int,
    cached_idx: int,
) -> float | None:
    query = context.queries[
        query_idx
    ]

    cached_query = context.queries[
        cached_idx
    ]

    if query == cached_query:
        return 1.0

    pair = (
        query,
        cached_query,
    )

    if pair not in context.known_pairs:
        return None

    return float(
        context.jaccard[pair]
    )


def _positive_cached_candidates(
    context: OnlineCacheContext,
    query_idx: int,
    cache_indices: list[int],
    relevance_threshold: float,
    relevance_operator: str,
) -> list[int]:
    """
    Known reusable results currently available in the cache.

    Used only to calculate reuse opportunities / oracle performance.
    """
    positive: list[int] = []

    for cached_idx in cache_indices:
        label = _pair_label(
            context=context,
            query_idx=query_idx,
            cached_idx=cached_idx,
            relevance_threshold=relevance_threshold,
            relevance_operator=relevance_operator,
        )

        if label == "positive":
            positive.append(
                cached_idx
            )

    return positive


# ============================================================
# FIFO cache helper
# ============================================================


def _insert_fifo(
    cache: OrderedDict[int, None],
    query_idx: int,
    capacity: int,
) -> None:
    """
    Insert an executed query result into the bounded FIFO cache.

    FIFO semantics:
      - insertion order determines eviction order
      - cache hits NEVER refresh or reorder entries
      - if K is exceeded, evict the oldest inserted entry
    """
    if capacity <= 0:
        return

    # A unique-query workload should normally prevent reinsertion of an
    # existing item, but keeping this guard makes the helper robust.
    if query_idx in cache:
        return

    cache[
        query_idx
    ] = None

    while len(cache) > capacity:
        cache.popitem(
            last=False
        )


# ============================================================
# Metric handling
# ============================================================


def _new_counts() -> dict[str, int | float]:
    return {
        "total_queries": 0,

        # Every cosine-threshold acceptance is a runtime semantic hit,
        # regardless of whether offline ground truth is known.
        "reuses": 0,
        "semantic_hits": 0,

        # Binary scored hits use only the deliberately constructed positive
        # and negative regions (J >= 0.8 and J <= 0.1 by default).
        "scored_reuses": 0,
        "scored_semantic_hits": 0,

        "correct_reuses": 0,
        "correct_semantic_hits": 0,

        "incorrect_reuses": 0,
        "false_semantic_hits": 0,

        # Known Jaccard in the gap between negative and positive thresholds.
        "intermediate_reuses": 0,
        "intermediate_semantic_hits": 0,

        # Selected pair has no stored anchor-candidate Jaccard annotation.
        "unknown_selected_reuses": 0,
        "unknown_selected_hits": 0,

        # Runtime misses. With the fixed-threshold policy these are exactly
        # the arrivals inserted into the FIFO cache.
        "executions": 0,
        "cache_misses": 0,
        "cache_insertions": 0,

        # Evaluation-only opportunity accounting based on known positives.
        "reusable_opportunities": 0,
        "missed_reuse_opportunities": 0,

        # Accepted hits for which some Jaccard value is available, including
        # intermediate pairs.
        "known_selected_reuses": 0,

        # Used to report mean answer-set overlap of known semantic hits.
        "known_hit_jaccard_sum": 0.0,
        "known_hit_jaccard_count": 0,
    }


_COUNT_KEYS = {
    "total_queries",
    "reuses",
    "semantic_hits",
    "scored_reuses",
    "scored_semantic_hits",
    "correct_reuses",
    "correct_semantic_hits",
    "incorrect_reuses",
    "false_semantic_hits",
    "intermediate_reuses",
    "intermediate_semantic_hits",
    "unknown_selected_reuses",
    "unknown_selected_hits",
    "executions",
    "cache_misses",
    "cache_insertions",
    "reusable_opportunities",
    "missed_reuse_opportunities",
    "known_selected_reuses",
    "known_hit_jaccard_sum",
    "known_hit_jaccard_count",
}


def _derive_metrics(
    counts: dict[str, int | float],
) -> dict[str, Any]:
    total = max(int(counts["total_queries"]), 1)

    correct = int(counts["correct_semantic_hits"])
    false = int(counts["false_semantic_hits"])
    intermediate = int(counts["intermediate_semantic_hits"])
    unknown = int(counts["unknown_selected_hits"])
    semantic_hits = int(counts["semantic_hits"])
    misses = int(counts["cache_misses"])
    opportunities = int(counts["reusable_opportunities"])

    # Precision deliberately follows the dataset's binary construction:
    # J >= positive_min versus J <= negative_max. Intermediate and unknown
    # accepted hits are reported separately rather than forced into a class.
    binary_scored = correct + false
    precision: float | None = (
        correct / binary_scored
        if binary_scored > 0
        else None
    )

    opportunity_recall = (
        correct / opportunities
        if opportunities > 0
        else 0.0
    )

    if precision is None or precision + opportunity_recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2.0
            * precision
            * opportunity_recall
            / (precision + opportunity_recall)
        )

    known_j_count = int(counts["known_hit_jaccard_count"])
    mean_jaccard_known_hits: float | None = (
        float(counts["known_hit_jaccard_sum"]) / known_j_count
        if known_j_count > 0
        else None
    )

    # Expected known answer-overlap delivered per incoming query. Unknown
    # selected pairs contribute no value because their Jaccard is unavailable.
    known_jaccard_utility = (
        float(counts["known_hit_jaccard_sum"]) / total
    )

    return {
        **counts,

        "precision": (
            None if precision is None else float(precision)
        ),
        "opportunity_recall": float(opportunity_recall),
        "f1": float(f1),

        # Runtime cache behavior.
        "semantic_hit_rate": float(semantic_hits / total),
        "reuse_rate": float(semantic_hits / total),
        "cache_miss_rate": float(misses / total),
        "execution_rate": float(misses / total),
        "cache_insertion_rate": float(misses / total),

        # Ground-truth quality of accepted hits.
        "correct_semantic_hit_rate": float(correct / total),
        "false_semantic_hit_rate": float(false / total),
        "intermediate_semantic_hit_rate": float(intermediate / total),
        "unknown_selection_rate": float(unknown / total),
        "known_selection_rate": float((semantic_hits - unknown) / total),
        "binary_scored_hit_rate": float(binary_scored / total),

        # Opportunity diagnostics. These are lower-bound/known-label metrics
        # when pairwise ground truth is incomplete.
        "opportunity_rate": float(opportunities / total),
        "reuse_efficiency": float(opportunity_recall),

        # Graded quality diagnostics.
        "mean_jaccard_known_hits": mean_jaccard_known_hits,
        "known_jaccard_utility": float(known_jaccard_utility),

        # Compatibility aliases used by older result summaries.
        "semantic_selection_rate": float(semantic_hits / total),
        "false_reuse_rate": float(false / total),
        "safe_execution_savings": float(correct / total),
        "hit_rate": float(semantic_hits / total),
        "false_hit_rate": float(false / total),
        "correct_hit_rate": float(correct / total),
        "execution_savings": float(correct / total),
        "recall": float(opportunity_recall),
    }


# ============================================================
# Stream generation
# ============================================================


def _make_stream_order(
    n_queries: int,
    seed: int,
    shuffle: bool,
) -> list[int]:
    """
    Create an online arrival ordering over the global unique workload.
    """
    if not shuffle:
        return list(
            range(n_queries)
        )

    rng = np.random.default_rng(
        seed
    )

    return (
        rng
        .permutation(n_queries)
        .tolist()
    )


# ============================================================
# Semantic-cache simulation
# ============================================================


def _simulate_semantic_stream(
    context: OnlineCacheContext,
    order: list[int],
    capacity: int,
    threshold: float,
    relevance_threshold: float,
    relevance_operator: str,
    negative_jaccard_max: float = 0.1,
) -> dict[str, Any]:
    """
    Replay one fixed-threshold semantic-cache workload.

    Runtime behavior:
        * nearest cached query is selected by cosine similarity only;
        * cosine >= threshold -> semantic hit/reuse, no insertion;
        * cosine < threshold  -> miss, insert incoming query into FIFO cache.

    Offline evaluation happens AFTER the runtime decision:
        * positive:      J >= positive threshold (normally 0.8)
        * intermediate:  negative_max < J < positive threshold
        * negative:      J <= negative threshold (normally 0.1)
        * unknown:       no stored pairwise Jaccard label

    Crucially, offline labels never alter cache state. If an unknown pair is
    accepted by cosine, it remains a runtime semantic hit; it is merely
    excluded from ground-truth quality scoring.
    """
    cache: OrderedDict[int, None] = OrderedDict()
    counts = _new_counts()

    for query_idx in order:
        counts["total_queries"] += 1
        cache_indices = list(cache.keys())

        # Empty cache: unavoidable miss and insertion.
        if not cache_indices:
            counts["executions"] += 1
            counts["cache_misses"] += 1
            counts["cache_insertions"] += 1
            _insert_fifo(cache, query_idx, capacity)
            continue

        # Evaluation-only: was a KNOWN positive already available?
        positives_in_cache = _positive_cached_candidates(
            context=context,
            query_idx=query_idx,
            cache_indices=cache_indices,
            relevance_threshold=relevance_threshold,
            relevance_operator=relevance_operator,
        )
        opportunity_exists = bool(positives_in_cache)
        if opportunity_exists:
            counts["reusable_opportunities"] += 1

        # Runtime decision uses cosine only.
        cache_array = np.asarray(cache_indices, dtype=np.int64)
        similarities = context.similarities[query_idx, cache_array]
        nearest_local_idx = int(np.argmax(similarities))
        cached_idx = cache_indices[nearest_local_idx]
        nearest_cosine = float(similarities[nearest_local_idx])

        if nearest_cosine < threshold:
            counts["executions"] += 1
            counts["cache_misses"] += 1
            counts["cache_insertions"] += 1
            if opportunity_exists:
                counts["missed_reuse_opportunities"] += 1
            _insert_fifo(cache, query_idx, capacity)
            continue

        # Cosine threshold accepted -> runtime semantic hit.
        counts["reuses"] += 1
        counts["semantic_hits"] += 1

        label = _pair_label(
            context=context,
            query_idx=query_idx,
            cached_idx=cached_idx,
            relevance_threshold=relevance_threshold,
            relevance_operator=relevance_operator,
            negative_jaccard_max=negative_jaccard_max,
        )

        jaccard = _pair_jaccard(
            context=context,
            query_idx=query_idx,
            cached_idx=cached_idx,
        )

        if jaccard is not None:
            counts["known_selected_reuses"] += 1
            counts["known_hit_jaccard_sum"] += float(jaccard)
            counts["known_hit_jaccard_count"] += 1

        if label == "positive":
            counts["scored_reuses"] += 1
            counts["scored_semantic_hits"] += 1
            counts["correct_reuses"] += 1
            counts["correct_semantic_hits"] += 1
            continue

        if label == "negative":
            counts["scored_reuses"] += 1
            counts["scored_semantic_hits"] += 1
            counts["incorrect_reuses"] += 1
            counts["false_semantic_hits"] += 1
            if opportunity_exists:
                counts["missed_reuse_opportunities"] += 1
            continue

        if label == "intermediate":
            counts["intermediate_reuses"] += 1
            counts["intermediate_semantic_hits"] += 1
            if opportunity_exists:
                counts["missed_reuse_opportunities"] += 1
            continue

        # Unknown accepted pair: runtime still reused. Do not insert it.
        counts["unknown_selected_reuses"] += 1
        counts["unknown_selected_hits"] += 1
        if opportunity_exists:
            counts["missed_reuse_opportunities"] += 1

    metrics = _derive_metrics(counts)
    metrics["threshold"] = float(threshold)
    metrics["capacity"] = int(capacity)
    return metrics


# ============================================================
# Exact-match baseline
# ============================================================


def _simulate_exact_match_stream(
    context: OnlineCacheContext,
    order: list[int],
    capacity: int,
) -> dict[str, Any]:
    """
    Conventional exact-query FIFO cache.

    The primary workload contains each query string once, so exact-match hits
    should normally be zero. This baseline documents that semantic hits arise
    from reuse across different query strings.
    """
    cache: OrderedDict[int, None] = OrderedDict()
    counts = _new_counts()

    for query_idx in order:
        counts["total_queries"] += 1
        query = context.queries[query_idx]

        exact_idx: int | None = None
        for cached_idx in cache.keys():
            if context.queries[cached_idx] == query:
                exact_idx = cached_idx
                break

        if exact_idx is not None:
            counts["reuses"] += 1
            counts["semantic_hits"] += 1
            counts["scored_reuses"] += 1
            counts["scored_semantic_hits"] += 1
            counts["correct_reuses"] += 1
            counts["correct_semantic_hits"] += 1
            counts["known_selected_reuses"] += 1
            counts["known_hit_jaccard_sum"] += 1.0
            counts["known_hit_jaccard_count"] += 1
            counts["reusable_opportunities"] += 1
        else:
            counts["executions"] += 1
            counts["cache_misses"] += 1
            counts["cache_insertions"] += 1
            _insert_fifo(cache, query_idx, capacity)

    metrics = _derive_metrics(counts)
    metrics["capacity"] = int(capacity)
    return metrics


# ============================================================
# Oracle baseline
# ============================================================


def _simulate_oracle_stream(
    context: OnlineCacheContext,
    order: list[int],
    capacity: int,
    relevance_threshold: float,
    relevance_operator: str,
) -> dict[str, Any]:
    """
    Non-deployable KNOWN-LABEL oracle upper bound.

    The oracle reuses whenever a stored Jaccard-positive query is currently in
    cache. Because arbitrary workload pairs may be unlabelled, this is an upper
    bound only with respect to the available labelled positives, not a complete
    answer-set oracle unless full pairwise ground truth is supplied.
    """
    cache: OrderedDict[int, None] = OrderedDict()
    counts = _new_counts()

    for query_idx in order:
        counts["total_queries"] += 1
        cache_indices = list(cache.keys())
        positives = _positive_cached_candidates(
            context=context,
            query_idx=query_idx,
            cache_indices=cache_indices,
            relevance_threshold=relevance_threshold,
            relevance_operator=relevance_operator,
        )

        if positives:
            counts["reusable_opportunities"] += 1
            counts["reuses"] += 1
            counts["semantic_hits"] += 1
            counts["scored_reuses"] += 1
            counts["scored_semantic_hits"] += 1
            counts["correct_reuses"] += 1
            counts["correct_semantic_hits"] += 1
            counts["known_selected_reuses"] += 1

            best_cached_idx = max(
                positives,
                key=lambda cached_idx: (
                    _pair_jaccard(
                        context=context,
                        query_idx=query_idx,
                        cached_idx=cached_idx,
                    )
                    or 0.0
                ),
            )
            best_j = _pair_jaccard(
                context=context,
                query_idx=query_idx,
                cached_idx=best_cached_idx,
            )
            if best_j is not None:
                counts["known_hit_jaccard_sum"] += float(best_j)
                counts["known_hit_jaccard_count"] += 1
        else:
            counts["executions"] += 1
            counts["cache_misses"] += 1
            counts["cache_insertions"] += 1
            _insert_fifo(cache, query_idx, capacity)

    metrics = _derive_metrics(counts)
    metrics["capacity"] = int(capacity)
    return metrics


# ============================================================
# Pool counts across stream orderings
# ============================================================


def _pool_stream_counts(
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    pooled = _new_counts()

    for run in runs:
        for key in _COUNT_KEYS:
            value = run.get(key, 0)
            if key == "known_hit_jaccard_sum":
                pooled[key] += float(value)
            else:
                pooled[key] += int(value)

    return _derive_metrics(pooled)


# ============================================================
# Aggregation across stream orderings
# ============================================================


def _aggregate_stream_metrics(
    runs: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    if not runs:
        return {}

    keys: set[str] = set()

    for run in runs:
        keys.update(
            run.keys()
        )

    aggregated: dict[
        str,
        dict[str, float | int],
    ] = {}

    for key in sorted(keys):
        if key in {
            "stream_seed",
            "capacity",
            "threshold",
        }:
            continue

        values: list[float] = []

        for run in runs:
            value = run.get(
                key
            )

            if isinstance(
                value,
                bool,
            ):
                continue

            if isinstance(
                value,
                (int, float),
            ):
                values.append(
                    float(value)
                )

        if not values:
            continue

        aggregated[key] = {
            "mean": float(
                np.mean(values)
            ),
            "std": float(
                np.std(
                    values,
                    ddof=0,
                )
            ),
            "n": len(
                values
            ),
        }

    return aggregated


# ============================================================
# Evaluate one cache capacity
# ============================================================


def _evaluate_capacity(
    context: OnlineCacheContext,
    stream_orders: list[
        tuple[
            int,
            list[int],
        ]
    ],
    capacity: int,
    threshold: float,
    relevance_threshold: float,
    relevance_operator: str,
    cache_cfg: dict[str, Any],
    negative_jaccard_max: float = 0.1,
) -> dict[str, Any]:
    include_exact = bool(
        cache_cfg.get(
            "include_exact_match_baseline",
            True,
        )
    )

    include_oracle = bool(
        cache_cfg.get(
            "include_oracle_baseline",
            True,
        )
    )

    semantic_runs: list[
        dict[str, Any]
    ] = []

    exact_runs: list[
        dict[str, Any]
    ] = []

    oracle_runs: list[
        dict[str, Any]
    ] = []

    for (
        stream_seed,
        order,
    ) in stream_orders:
        semantic = (
            _simulate_semantic_stream(
                context=context,
                order=order,
                capacity=capacity,
                threshold=threshold,
                relevance_threshold=relevance_threshold,
                relevance_operator=relevance_operator,
                negative_jaccard_max=negative_jaccard_max,
            )
        )

        semantic[
            "stream_seed"
        ] = int(
            stream_seed
        )

        semantic_runs.append(
            semantic
        )

        if include_exact:
            exact = (
                _simulate_exact_match_stream(
                    context=context,
                    order=order,
                    capacity=capacity,
                )
            )

            exact[
                "stream_seed"
            ] = int(
                stream_seed
            )

            exact_runs.append(
                exact
            )

        if include_oracle:
            oracle = (
                _simulate_oracle_stream(
                    context=context,
                    order=order,
                    capacity=capacity,
                    relevance_threshold=relevance_threshold,
                    relevance_operator=relevance_operator,
                )
            )

            oracle[
                "stream_seed"
            ] = int(
                stream_seed
            )

            oracle_runs.append(
                oracle
            )

    result: dict[str, Any] = {
        "semantic": {
            "aggregate": (
                _aggregate_stream_metrics(
                    semantic_runs
                )
            ),
            "per_stream": semantic_runs,
        }
    }

    if include_exact:
        result[
            "exact_match"
        ] = {
            "aggregate": (
                _aggregate_stream_metrics(
                    exact_runs
                )
            ),
            "per_stream": exact_runs,
        }

    if include_oracle:
        result[
            "oracle"
        ] = {
            "aggregate": (
                _aggregate_stream_metrics(
                    oracle_runs
                )
            ),
            "per_stream": oracle_runs,
        }

    return result


# ============================================================
# Fixed-threshold sensitivity
# ============================================================


def _evaluate_threshold_sensitivity(
    context: OnlineCacheContext,
    stream_orders: list[tuple[int, list[int]]],
    capacity: int,
    thresholds: list[float],
    relevance_threshold: float,
    relevance_operator: str,
    negative_jaccard_max: float,
) -> list[dict[str, Any]]:
    """Return pooled semantic-cache metrics for a predefined threshold grid.

    This is a robustness analysis only. No threshold is selected from the test
    sweep; the headline threshold remains the fixed a-priori configuration.
    """
    out: list[dict[str, Any]] = []

    for threshold in thresholds:
        runs: list[dict[str, Any]] = []
        for stream_seed, order in stream_orders:
            metrics = _simulate_semantic_stream(
                context=context,
                order=order,
                capacity=capacity,
                threshold=float(threshold),
                relevance_threshold=relevance_threshold,
                relevance_operator=relevance_operator,
                negative_jaccard_max=negative_jaccard_max,
            )
            metrics["stream_seed"] = int(stream_seed)
            runs.append(metrics)

        pooled = _pool_stream_counts(runs)
        pooled["threshold"] = float(threshold)
        out.append(pooled)

    return out


# ============================================================
# Public entry point
# ============================================================


def evaluate_semantic_cache(
    model: HFEmbeddingModel,
    validation_records: list[QueryRecord],
    test_records: list[QueryRecord],
    eval_cfg: dict[str, Any],
    cache_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate a fixed-threshold FIFO semantic cache.

    Headline question:
        At the same cache size and the same cosine acceptance threshold, does
        result-aware fine-tuning produce more correct semantic reuse and/or
        less false reuse?

    The default runtime rule is intentionally simple and fixed a priori:

        nearest-cache cosine >= 0.8  -> reuse cached result
        nearest-cache cosine <  0.8  -> miss and insert incoming result

    Jaccard is inspected only after a hit for offline evaluation.
    """
    if not cache_cfg.get("enabled", True):
        return {"definition": {"mode": "disabled"}}

    positive_jaccard_min = float(
        cache_cfg.get(
            "positive_jaccard_min",
            eval_cfg.get("relevance_threshold", 0.8),
        )
    )
    negative_jaccard_max = float(
        cache_cfg.get("negative_jaccard_max", 0.1)
    )
    relevance_operator = str(
        eval_cfg.get("relevance_operator", "ge")
    )

    if negative_jaccard_max >= positive_jaccard_min:
        raise ValueError(
            "semantic_cache.negative_jaccard_max must be smaller than "
            "semantic_cache.positive_jaccard_min"
        )

    cosine_threshold = float(
        cache_cfg.get(
            "cosine_threshold",
            cache_cfg.get("threshold", 0.8),
        )
    )
    if not -1.0 <= cosine_threshold <= 1.0:
        raise ValueError(
            "semantic_cache.cosine_threshold must lie in [-1, 1]"
        )

    encode_batch_size = int(
        cache_cfg.get(
            "encode_batch_size",
            eval_cfg.get("encode_batch_size", 64),
        )
    )

    capacities = [
        int(value)
        for value in cache_cfg.get(
            "capacities", [32, 64, 128, 256]
        )
    ]
    if not capacities or any(capacity <= 0 for capacity in capacities):
        raise ValueError(
            "semantic_cache.capacities must contain positive integers"
        )

    primary_capacity = int(cache_cfg.get("primary_capacity", 128))
    if primary_capacity <= 0:
        raise ValueError("semantic_cache.primary_capacity must be positive")
    if primary_capacity not in capacities:
        capacities = sorted(set(capacities + [primary_capacity]))

    stream_seeds = [
        int(seed)
        for seed in cache_cfg.get(
            "stream_seeds", [100, 101, 102, 103, 104]
        )
    ]
    if not stream_seeds:
        raise ValueError("semantic_cache.stream_seeds cannot be empty")

    shuffle_streams = bool(cache_cfg.get("shuffle_streams", True))

    validation_context = _build_online_context(
        model=model,
        records=validation_records,
        batch_size=encode_batch_size,
    )
    test_context = _build_online_context(
        model=model,
        records=test_records,
        batch_size=encode_batch_size,
    )

    validation_orders = [
        (
            seed,
            _make_stream_order(
                len(validation_context.queries), seed, shuffle_streams
            ),
        )
        for seed in stream_seeds
    ]
    test_orders = [
        (
            seed,
            _make_stream_order(len(test_context.queries), seed, shuffle_streams),
        )
        for seed in stream_seeds
    ]

    # Validation is diagnostic only. It does NOT choose the threshold.
    validation_runs: list[dict[str, Any]] = []
    for stream_seed, order in validation_orders:
        metrics = _simulate_semantic_stream(
            context=validation_context,
            order=order,
            capacity=primary_capacity,
            threshold=cosine_threshold,
            relevance_threshold=positive_jaccard_min,
            relevance_operator=relevance_operator,
            negative_jaccard_max=negative_jaccard_max,
        )
        metrics["stream_seed"] = int(stream_seed)
        validation_runs.append(metrics)
    validation_pooled = _pool_stream_counts(validation_runs)
    validation_pooled["threshold"] = float(cosine_threshold)

    capacity_results: dict[str, Any] = {}
    for capacity in capacities:
        capacity_results[str(capacity)] = {
            "test": _evaluate_capacity(
                context=test_context,
                stream_orders=test_orders,
                capacity=capacity,
                threshold=cosine_threshold,
                relevance_threshold=positive_jaccard_min,
                relevance_operator=relevance_operator,
                cache_cfg=cache_cfg,
                negative_jaccard_max=negative_jaccard_max,
            )
        }

    output: dict[str, Any] = {
        "definition": {
            "mode": "offline fixed-threshold semantic-cache replay",
            "workload_universe": (
                "all unique queries in the split: anchors plus positive and "
                "negative candidates"
            ),
            "workload_unique": True,
            "stream_order": (
                "seeded random permutation of the global unique query set"
                if shuffle_streams
                else "dataset-derived global query order"
            ),
            "cache_policy": "FIFO",
            "semantic_selection": (
                "nearest query currently in cache by cosine similarity"
            ),
            "runtime_reuse_rule": (
                f"semantic hit iff nearest-cache cosine >= {cosine_threshold}"
            ),
            "threshold_mode": "fixed a priori; no validation calibration",
            "cosine_threshold": float(cosine_threshold),
            "ground_truth_labels": {
                "positive": f"stored pair with J >= {positive_jaccard_min}",
                "intermediate": (
                    f"stored pair with {negative_jaccard_max} < J < "
                    f"{positive_jaccard_min}"
                ),
                "negative": f"stored pair with J <= {negative_jaccard_max}",
                "unknown": (
                    "pair has no stored anchor-candidate Jaccard annotation"
                ),
            },
            "runtime_state_rule": (
                "offline labels never alter cache state: every cosine-accepted "
                "pair is reused and not inserted, including unknown pairs"
            ),
            "result_ground_truth_usage": (
                "Jaccard is used only after runtime selection for offline "
                "quality evaluation and by the non-deployable known-label oracle"
            ),
            "primary_capacity": primary_capacity,
            "capacities": capacities,
            "stream_seeds": stream_seeds,
            "baselines": {
                "exact_match": bool(
                    cache_cfg.get("include_exact_match_baseline", True)
                ),
                "known_label_oracle": bool(
                    cache_cfg.get("include_oracle_baseline", True)
                ),
            },
        },

        # Backward-compatible field name. It is fixed, not selected.
        "selected_threshold": float(cosine_threshold),
        "fixed_threshold": float(cosine_threshold),

        "validation": {
            "primary_capacity": primary_capacity,
            "fixed_threshold_pooled": validation_pooled,
            # Keep the old key so the summary writer can still discover the
            # headline operating point without special handling.
            "selected_operating_point": validation_pooled,
        },
        "capacities": capacity_results,
        "validation_examples": len(validation_context.queries),
        "test_examples": len(test_context.queries),
        "validation_known_pair_count": len(validation_context.known_pairs),
        "test_known_pair_count": len(test_context.known_pairs),
    }

    # Optional old-style robustness sweep. The fixed 0.8 (or configured)
    # threshold remains the headline; this sweep is never used to pick it.
    if bool(cache_cfg.get("run_threshold_sensitivity", False)):
        thresholds = [
            float(value)
            for value in cache_cfg.get(
                "thresholds",
                [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            )
        ]
        output["threshold_sensitivity"] = {
            "split": "test",
            "capacity": primary_capacity,
            "note": (
                "robustness sweep only; headline threshold is fixed a priori"
            ),
            "points": _evaluate_threshold_sensitivity(
                context=test_context,
                stream_orders=test_orders,
                capacity=primary_capacity,
                thresholds=thresholds,
                relevance_threshold=positive_jaccard_min,
                relevance_operator=relevance_operator,
                negative_jaccard_max=negative_jaccard_max,
            ),
        }

    return output

