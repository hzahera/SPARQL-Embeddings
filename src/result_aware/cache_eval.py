from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .data import QueryRecord
from .model import HFEmbeddingModel


# ============================================================
# Data structures
# ============================================================


@dataclass(frozen=True)
class RQ3GroundTruth:
    """
    Complete TEST-only binary ground truth for RQ3.

    The RQ3 builder evaluates every unordered pair of unique test queries and
    stores only the two deliberately scored regions:

        positive_pairs: J >= positive_jaccard_min
        negative_pairs: J <= negative_jaccard_max

    A distinct pair absent from both lists is therefore NOT unknown. It is a
    deliberately excluded middle pair:

        negative_jaccard_max < J < positive_jaccard_min

    Exact Jaccard values are available only for the stored positive/negative
    pairs because the middle pairs were intentionally not written to the RQ3
    JSON.
    """

    path: str
    positive_pairs: dict[tuple[int, int], float]
    negative_pairs: dict[tuple[int, int], float]
    positive_jaccard_min: float
    negative_jaccard_max: float
    all_pair_count: int
    positive_pair_count: int
    negative_pair_count: int
    excluded_middle_pair_count: int


@dataclass(frozen=True)
class OnlineCacheContext:
    """
    Precomputed information for one RQ3 TEST workload.

    queries:
        Global unique test-query universe:
        anchors + all attached candidate queries.

    similarities:
        Pairwise cosine-similarity matrix between all unique test queries.

    query_to_idx:
        Mapping from query text to row/column in the similarity matrix.

    ground_truth:
        Complete test-only RQ3 pair relation.
    """

    queries: list[str]
    similarities: np.ndarray
    query_to_idx: dict[str, int]
    ground_truth: RQ3GroundTruth


# ============================================================
# Query helpers
# ============================================================


def _normalize_query(q: str) -> str:
    """
    Must match scripts/build_rq3_from_endpoint.py.
    """
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    return q


def _query_hash(q: str) -> str:
    return hashlib.sha256(
        _normalize_query(q).encode("utf-8")
    ).hexdigest()


def _candidate_query(candidate: Any) -> str:
    """
    Obtain the SPARQL string from a candidate object.
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
    Global unique workload:
        anchors + positive candidates + negative candidates.

    Each textual query appears exactly once.
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
# RQ3 file discovery / validation
# ============================================================


@lru_cache(maxsize=8)
def _read_json_cached(path_str: str) -> dict[str, Any]:
    path = Path(path_str)

    if not path.exists():
        raise FileNotFoundError(
            f"RQ3 ground-truth file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(
            f"RQ3 JSON root must be an object: {path}"
        )

    return obj


def _configured_rq3_paths(
    cache_cfg: dict[str, Any],
) -> list[Path]:
    """
    Supported configuration forms:

    Single-dataset run:
        rq3_test_file: datasets/.../dbpedia-test-rq3.json

    Multi-dataset run:
        rq3_test_files:
          dbpedia: datasets/dbpedia-splits/dbpedia-test-rq3.json
          wikidata: datasets/wikidata-splits/wikidata-test-rq3.json

    A list is also accepted:
        rq3_test_files:
          - path/to/dbpedia-test-rq3.json
          - path/to/wikidata-test-rq3.json

    When multiple paths are configured, this evaluator identifies the correct
    one by exact query-hash-set equality with the current test workload.
    """
    paths: list[Path] = []

    single = cache_cfg.get("rq3_test_file")
    if isinstance(single, str) and single.strip():
        paths.append(Path(single))

    multiple = cache_cfg.get("rq3_test_files")

    if isinstance(multiple, dict):
        for value in multiple.values():
            if isinstance(value, str) and value.strip():
                paths.append(Path(value))

    elif isinstance(multiple, (list, tuple)):
        for value in multiple:
            if isinstance(value, str) and value.strip():
                paths.append(Path(value))

    # Preserve order while removing duplicates.
    out: list[Path] = []
    seen: set[str] = set()

    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)

    if not out:
        raise ValueError(
            "RQ3 semantic-cache evaluation requires a test-only RQ3 JSON. "
            "Configure semantic_cache.rq3_test_file or "
            "semantic_cache.rq3_test_files."
        )

    return out


def _rq3_query_hash_set(raw: dict[str, Any], path: Path) -> set[str]:
    rows = raw.get("queries")

    if not isinstance(rows, list):
        raise ValueError(
            f"RQ3 file has no list-valued 'queries': {path}"
        )

    hashes: set[str] = set()

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(
                f"Invalid query entry in RQ3 file: {path}"
            )

        qh = row.get("query_hash")

        # The builder writes query_hash. Fallback to query text makes the
        # evaluator robust to an equivalent hand-built file.
        if not isinstance(qh, str) or not qh:
            query = row.get("query")
            if not isinstance(query, str):
                raise ValueError(
                    f"RQ3 query row lacks query_hash/query: {path}"
                )
            qh = _query_hash(query)

        if qh in hashes:
            raise ValueError(
                f"Duplicate query_hash in RQ3 file {path}: {qh}"
            )

        hashes.add(qh)

    return hashes


def _resolve_rq3_path(
    cache_cfg: dict[str, Any],
    workload_queries: list[str],
) -> Path:
    """
    Choose the RQ3 file whose query universe exactly matches this test split.

    This allows the unchanged experiment runner to evaluate DBpedia and
    Wikidata in the same command without needing to pass the dataset name into
    evaluate_semantic_cache().
    """
    expected = {
        _query_hash(query)
        for query in workload_queries
    }

    candidates = _configured_rq3_paths(cache_cfg)

    exact_matches: list[Path] = []
    mismatch_summaries: list[str] = []

    for path in candidates:
        raw = _read_json_cached(str(path))
        actual = _rq3_query_hash_set(raw, path)

        if actual == expected:
            exact_matches.append(path)
        else:
            missing = len(expected - actual)
            extra = len(actual - expected)
            mismatch_summaries.append(
                f"{path}: missing={missing}, extra={extra}, "
                f"rq3_queries={len(actual)}, workload_queries={len(expected)}"
            )

    if len(exact_matches) == 1:
        return exact_matches[0]

    if len(exact_matches) > 1:
        raise ValueError(
            "More than one configured RQ3 file exactly matches the current "
            f"test workload: {[str(p) for p in exact_matches]}"
        )

    details = "\n  ".join(mismatch_summaries)

    raise ValueError(
        "None of the configured RQ3 files matches the current TEST query "
        "universe exactly.\n"
        f"  {details}"
    )


def _unordered_pair(i: int, j: int) -> tuple[int, int]:
    if i == j:
        return (i, j)
    return (i, j) if i < j else (j, i)


def _load_rq3_ground_truth(
    path: Path,
    workload_queries: list[str],
    expected_positive_min: float,
    expected_negative_max: float,
) -> RQ3GroundTruth:
    raw = _read_json_cached(str(path))
    metadata = raw.get("metadata", {})

    if not isinstance(metadata, dict):
        raise ValueError(
            f"RQ3 metadata must be an object: {path}"
        )

    rq3_positive_min = float(
        metadata.get(
            "positive_jaccard_min",
            expected_positive_min,
        )
    )
    rq3_negative_max = float(
        metadata.get(
            "negative_jaccard_max",
            expected_negative_max,
        )
    )

    if not np.isclose(
        rq3_positive_min,
        expected_positive_min,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            f"RQ3 positive threshold mismatch for {path}: "
            f"file={rq3_positive_min}, config={expected_positive_min}"
        )

    if not np.isclose(
        rq3_negative_max,
        expected_negative_max,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            f"RQ3 negative threshold mismatch for {path}: "
            f"file={rq3_negative_max}, config={expected_negative_max}"
        )

    query_rows = raw.get("queries")

    if not isinstance(query_rows, list):
        raise ValueError(
            f"RQ3 file has no list-valued 'queries': {path}"
        )

    # Current workload index by canonical query hash.
    workload_hash_to_idx: dict[str, int] = {}

    for idx, query in enumerate(workload_queries):
        qh = _query_hash(query)

        if qh in workload_hash_to_idx:
            raise ValueError(
                "Two current workload queries collapse to the same normalized "
                f"hash: {qh}"
            )

        workload_hash_to_idx[qh] = idx

    # Map RQ3 integer IDs -> current context indices.
    rq3_id_to_context_idx: dict[int, int] = {}
    rq3_hashes: set[str] = set()

    for row in query_rows:
        if not isinstance(row, dict):
            raise ValueError(
                f"Invalid query row in {path}"
            )

        if "id" not in row:
            raise ValueError(
                f"RQ3 query row is missing 'id': {path}"
            )

        rq3_id = int(row["id"])

        if rq3_id in rq3_id_to_context_idx:
            raise ValueError(
                f"Duplicate RQ3 query id {rq3_id} in {path}"
            )

        qh = row.get("query_hash")

        if not isinstance(qh, str) or not qh:
            query = row.get("query")
            if not isinstance(query, str):
                raise ValueError(
                    f"RQ3 query id {rq3_id} lacks query_hash/query in {path}"
                )
            qh = _query_hash(query)

        rq3_hashes.add(qh)

        if qh not in workload_hash_to_idx:
            raise ValueError(
                f"RQ3 query id {rq3_id} is not present in current TEST "
                f"workload: {path}"
            )

        rq3_id_to_context_idx[rq3_id] = (
            workload_hash_to_idx[qh]
        )

    expected_hashes = set(workload_hash_to_idx)

    if rq3_hashes != expected_hashes:
        raise ValueError(
            f"RQ3/current-test query mismatch for {path}: "
            f"missing={len(expected_hashes - rq3_hashes)}, "
            f"extra={len(rq3_hashes - expected_hashes)}"
        )

    positive_pairs: dict[tuple[int, int], float] = {}
    negative_pairs: dict[tuple[int, int], float] = {}

    def load_pairs(
        field: str,
        target: dict[tuple[int, int], float],
        kind: str,
    ) -> None:
        rows = raw.get(field)

        if not isinstance(rows, list):
            raise ValueError(
                f"RQ3 field '{field}' must be a list: {path}"
            )

        for row in rows:
            if (
                not isinstance(row, list)
                or len(row) != 3
            ):
                raise ValueError(
                    f"Each {field} entry must be [id1, id2, jaccard]: {path}"
                )

            rq3_i = int(row[0])
            rq3_j = int(row[1])
            jac = float(row[2])

            if rq3_i not in rq3_id_to_context_idx:
                raise ValueError(
                    f"Unknown RQ3 query id {rq3_i} in {field}: {path}"
                )

            if rq3_j not in rq3_id_to_context_idx:
                raise ValueError(
                    f"Unknown RQ3 query id {rq3_j} in {field}: {path}"
                )

            i = rq3_id_to_context_idx[rq3_i]
            j = rq3_id_to_context_idx[rq3_j]

            if i == j:
                raise ValueError(
                    f"RQ3 pair lists must not contain self-pairs: {path}"
                )

            key = _unordered_pair(i, j)

            if key in target:
                raise ValueError(
                    f"Duplicate pair {key} in {field}: {path}"
                )

            if kind == "positive":
                if jac + 1e-12 < expected_positive_min:
                    raise ValueError(
                        f"Positive pair below positive threshold in {path}: "
                        f"J={jac}"
                    )
            else:
                if jac - 1e-12 > expected_negative_max:
                    raise ValueError(
                        f"Negative pair above negative threshold in {path}: "
                        f"J={jac}"
                    )

            target[key] = jac

    load_pairs(
        "positive_pairs",
        positive_pairs,
        "positive",
    )
    load_pairs(
        "negative_pairs",
        negative_pairs,
        "negative",
    )

    overlap = set(positive_pairs) & set(negative_pairs)

    if overlap:
        example = next(iter(overlap))
        raise ValueError(
            f"A pair appears as both positive and negative in {path}: "
            f"{example}"
        )

    n = len(workload_queries)
    expected_all_pair_count = n * (n - 1) // 2

    file_all_pair_count = int(
        metadata.get(
            "all_unordered_pair_count",
            metadata.get(
                "all_pair_count",
                expected_all_pair_count,
            ),
        )
    )

    if file_all_pair_count != expected_all_pair_count:
        raise ValueError(
            f"RQ3 all-pair count mismatch for {path}: "
            f"file={file_all_pair_count}, expected={expected_all_pair_count}"
        )

    excluded_middle_pair_count = int(
        metadata.get(
            "excluded_intermediate_pair_count",
            expected_all_pair_count
            - len(positive_pairs)
            - len(negative_pairs),
        )
    )

    accounted = (
        len(positive_pairs)
        + len(negative_pairs)
        + excluded_middle_pair_count
    )

    if accounted != expected_all_pair_count:
        raise ValueError(
            f"RQ3 pair accounting is incomplete for {path}: "
            f"positive={len(positive_pairs)}, "
            f"negative={len(negative_pairs)}, "
            f"excluded_middle={excluded_middle_pair_count}, "
            f"sum={accounted}, expected={expected_all_pair_count}. "
            "Without complete pair accounting, an absent pair could not be "
            "safely interpreted as the deliberately excluded middle region."
        )

    return RQ3GroundTruth(
        path=str(path),
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
        positive_jaccard_min=float(expected_positive_min),
        negative_jaccard_max=float(expected_negative_max),
        all_pair_count=int(expected_all_pair_count),
        positive_pair_count=len(positive_pairs),
        negative_pair_count=len(negative_pairs),
        excluded_middle_pair_count=int(excluded_middle_pair_count),
    )


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

    encode_fn = getattr(model, "encode", None)

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

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(norms, 1e-12)
    embeddings = embeddings / norms

    return embeddings


def _build_online_context(
    model: HFEmbeddingModel,
    records: list[QueryRecord],
    batch_size: int,
    cache_cfg: dict[str, Any],
    positive_jaccard_min: float,
    negative_jaccard_max: float,
) -> OnlineCacheContext:
    queries = _global_unique_queries(records)

    if not queries:
        raise ValueError(
            "RQ3 test workload contains no queries."
        )

    rq3_path = _resolve_rq3_path(
        cache_cfg=cache_cfg,
        workload_queries=queries,
    )

    ground_truth = _load_rq3_ground_truth(
        path=rq3_path,
        workload_queries=queries,
        expected_positive_min=positive_jaccard_min,
        expected_negative_max=negative_jaccard_max,
    )

    query_to_idx = {
        query: idx
        for idx, query in enumerate(queries)
    }

    embeddings = _encode_queries(
        model=model,
        queries=queries,
        batch_size=batch_size,
    )

    similarities = embeddings @ embeddings.T

    similarities = np.clip(
        similarities,
        -1.0,
        1.0,
    )

    return OnlineCacheContext(
        queries=queries,
        similarities=similarities,
        query_to_idx=query_to_idx,
        ground_truth=ground_truth,
    )


# ============================================================
# Complete RQ3 pair labels
# ============================================================


def _pair_label(
    context: OnlineCacheContext,
    query_idx: int,
    cached_idx: int,
) -> str:
    """
    Complete offline classification.

    Returns:
        positive
        negative
        excluded_middle

    There is NO unknown state because the RQ3 builder evaluated every
    unordered test-query pair and verified complete pair accounting.
    """
    if query_idx == cached_idx:
        return "positive"

    key = _unordered_pair(
        query_idx,
        cached_idx,
    )

    if key in context.ground_truth.positive_pairs:
        return "positive"

    if key in context.ground_truth.negative_pairs:
        return "negative"

    return "excluded_middle"


def _pair_jaccard(
    context: OnlineCacheContext,
    query_idx: int,
    cached_idx: int,
) -> float | None:
    """
    Exact Jaccard when it was stored in the binary RQ3 JSON.

    For deliberately omitted middle-region pairs the exact Jaccard is not
    available, so this returns None.
    """
    if query_idx == cached_idx:
        return 1.0

    key = _unordered_pair(
        query_idx,
        cached_idx,
    )

    if key in context.ground_truth.positive_pairs:
        return float(
            context.ground_truth.positive_pairs[key]
        )

    if key in context.ground_truth.negative_pairs:
        return float(
            context.ground_truth.negative_pairs[key]
        )

    return None


def _positive_cached_candidates(
    context: OnlineCacheContext,
    query_idx: int,
    cache_indices: list[int],
) -> list[int]:
    """
    Exact reusable-result opportunities currently present in cache.

    Unlike the previous evaluator, this is no longer a lower bound based only
    on anchor-candidate annotations. Every positive test pair is represented
    in the RQ3 ground-truth file.
    """
    positive: list[int] = []

    for cached_idx in cache_indices:
        if _pair_label(
            context,
            query_idx,
            cached_idx,
        ) == "positive":
            positive.append(cached_idx)

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
    FIFO:
      - misses insert
      - hits never refresh/reorder
      - oldest inserted item is evicted first
    """
    if capacity <= 0:
        return

    if query_idx in cache:
        return

    cache[query_idx] = None

    while len(cache) > capacity:
        cache.popitem(last=False)


# ============================================================
# Metric handling
# ============================================================


def _new_counts() -> dict[str, int | float]:
    return {
        "total_queries": 0,

        # Runtime cosine-threshold acceptances.
        "reuses": 0,
        "semantic_hits": 0,

        # Binary RQ3 scored hits.
        "scored_reuses": 0,
        "scored_semantic_hits": 0,

        # Positive-region selected hits.
        "correct_reuses": 0,
        "correct_semantic_hits": 0,

        # Negative-region selected hits.
        "incorrect_reuses": 0,
        "false_semantic_hits": 0,

        # Pair was evaluated by the RQ3 builder but intentionally omitted
        # because negative_max < J < positive_min.
        "excluded_middle_reuses": 0,
        "excluded_middle_hits": 0,

        # Compatibility aliases for earlier summary code.
        "intermediate_reuses": 0,
        "intermediate_semantic_hits": 0,

        # With complete RQ3 pair coverage these MUST remain zero.
        "unknown_selected_reuses": 0,
        "unknown_selected_hits": 0,

        # Runtime misses/executions/insertions.
        "executions": 0,
        "cache_misses": 0,
        "cache_insertions": 0,

        # Exact positive-opportunity accounting.
        "reusable_opportunities": 0,
        "missed_reuse_opportunities": 0,

        # Selected hits with exact J stored (positive + negative).
        "known_selected_reuses": 0,
        "known_hit_jaccard_sum": 0.0,
        "known_hit_jaccard_count": 0,
    }


_COUNT_KEYS = set(_new_counts().keys())


def _derive_metrics(
    counts: dict[str, int | float],
) -> dict[str, Any]:
    total = max(
        int(counts["total_queries"]),
        1,
    )

    correct = int(
        counts["correct_semantic_hits"]
    )
    false = int(
        counts["false_semantic_hits"]
    )
    excluded = int(
        counts["excluded_middle_hits"]
    )
    unknown = int(
        counts["unknown_selected_hits"]
    )
    semantic_hits = int(
        counts["semantic_hits"]
    )
    misses = int(
        counts["cache_misses"]
    )
    opportunities = int(
        counts["reusable_opportunities"]
    )

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

    if (
        precision is None
        or precision + opportunity_recall == 0
    ):
        f1 = 0.0
    else:
        f1 = (
            2.0
            * precision
            * opportunity_recall
            / (
                precision
                + opportunity_recall
            )
        )

    known_j_count = int(
        counts["known_hit_jaccard_count"]
    )

    mean_jaccard_known_hits: float | None = (
        float(
            counts["known_hit_jaccard_sum"]
        )
        / known_j_count
        if known_j_count > 0
        else None
    )

    known_jaccard_utility = (
        float(
            counts["known_hit_jaccard_sum"]
        )
        / total
    )

    return {
        **counts,

        # Binary quality over the intentionally scored regions only.
        "precision": (
            None
            if precision is None
            else float(precision)
        ),
        "opportunity_recall": float(
            opportunity_recall
        ),
        "f1": float(f1),

        # Runtime cache behavior.
        "semantic_hit_rate": float(
            semantic_hits / total
        ),
        "reuse_rate": float(
            semantic_hits / total
        ),
        "cache_miss_rate": float(
            misses / total
        ),
        "execution_rate": float(
            misses / total
        ),
        "cache_insertion_rate": float(
            misses / total
        ),

        # Offline quality of accepted hits.
        "correct_semantic_hit_rate": float(
            correct / total
        ),
        "false_semantic_hit_rate": float(
            false / total
        ),
        "excluded_middle_hit_rate": float(
            excluded / total
        ),

        # Compatibility alias: the middle region is known by construction,
        # but exact J was intentionally omitted from the RQ3 JSON.
        "intermediate_semantic_hit_rate": float(
            excluded / total
        ),

        # Complete RQ3 coverage means there are no genuinely unknown pairs.
        "unknown_selection_rate": float(
            unknown / total
        ),

        # Exact-J selection rate: positive + negative stored pairs.
        "known_selection_rate": float(
            binary_scored / total
        ),
        "binary_scored_hit_rate": float(
            binary_scored / total
        ),

        # Exact positive-opportunity diagnostics.
        "opportunity_rate": float(
            opportunities / total
        ),
        "reuse_efficiency": float(
            opportunity_recall
        ),

        # Exact J is available for positive/negative hits only.
        "mean_jaccard_known_hits": (
            mean_jaccard_known_hits
        ),
        "mean_jaccard_scored_hits": (
            mean_jaccard_known_hits
        ),
        "known_jaccard_utility": float(
            known_jaccard_utility
        ),
        "scored_jaccard_utility": float(
            known_jaccard_utility
        ),

        # Compatibility aliases.
        "semantic_selection_rate": float(
            semantic_hits / total
        ),
        "false_reuse_rate": float(
            false / total
        ),
        "safe_execution_savings": float(
            correct / total
        ),
        "hit_rate": float(
            semantic_hits / total
        ),
        "false_hit_rate": float(
            false / total
        ),
        "correct_hit_rate": float(
            correct / total
        ),
        "execution_savings": float(
            correct / total
        ),
        "recall": float(
            opportunity_recall
        ),
    }


# ============================================================
# Stream generation
# ============================================================


def _make_stream_order(
    n_queries: int,
    seed: int,
    shuffle: bool,
) -> list[int]:
    if not shuffle:
        return list(range(n_queries))

    rng = np.random.default_rng(seed)

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
) -> dict[str, Any]:
    """
    Replay one semantic-cache stream.

    Runtime:
        nearest cached query selected by cosine only
        cosine >= threshold -> hit/reuse; no insertion
        cosine < threshold  -> miss/execution; insert into FIFO

    Offline after a hit:
        positive        -> correct reuse
        negative        -> false reuse
        excluded_middle -> evaluated pair in the omitted Jaccard region

    Ground truth NEVER changes cache state.
    """
    cache: OrderedDict[int, None] = OrderedDict()
    counts = _new_counts()

    for query_idx in order:
        counts["total_queries"] += 1
        cache_indices = list(cache.keys())

        # Empty cache: unavoidable miss.
        if not cache_indices:
            counts["executions"] += 1
            counts["cache_misses"] += 1
            counts["cache_insertions"] += 1

            _insert_fifo(
                cache,
                query_idx,
                capacity,
            )
            continue

        # Evaluation only: does a true positive result already exist?
        positives_in_cache = (
            _positive_cached_candidates(
                context=context,
                query_idx=query_idx,
                cache_indices=cache_indices,
            )
        )

        opportunity_exists = bool(
            positives_in_cache
        )

        if opportunity_exists:
            counts[
                "reusable_opportunities"
            ] += 1

        # Runtime decision: cosine ONLY.
        cache_array = np.asarray(
            cache_indices,
            dtype=np.int64,
        )

        similarities = (
            context.similarities[
                query_idx,
                cache_array,
            ]
        )

        nearest_local_idx = int(
            np.argmax(similarities)
        )

        cached_idx = cache_indices[
            nearest_local_idx
        ]

        nearest_cosine = float(
            similarities[
                nearest_local_idx
            ]
        )

        # Miss -> execute incoming query and insert its result.
        if nearest_cosine < threshold:
            counts["executions"] += 1
            counts["cache_misses"] += 1
            counts["cache_insertions"] += 1

            if opportunity_exists:
                counts[
                    "missed_reuse_opportunities"
                ] += 1

            _insert_fifo(
                cache,
                query_idx,
                capacity,
            )
            continue

        # Hit -> runtime reuses cached result. No insertion.
        counts["reuses"] += 1
        counts["semantic_hits"] += 1

        label = _pair_label(
            context=context,
            query_idx=query_idx,
            cached_idx=cached_idx,
        )

        jac = _pair_jaccard(
            context=context,
            query_idx=query_idx,
            cached_idx=cached_idx,
        )

        if jac is not None:
            counts[
                "known_selected_reuses"
            ] += 1
            counts[
                "known_hit_jaccard_sum"
            ] += float(jac)
            counts[
                "known_hit_jaccard_count"
            ] += 1

        if label == "positive":
            counts["scored_reuses"] += 1
            counts[
                "scored_semantic_hits"
            ] += 1
            counts["correct_reuses"] += 1
            counts[
                "correct_semantic_hits"
            ] += 1
            continue

        if label == "negative":
            counts["scored_reuses"] += 1
            counts[
                "scored_semantic_hits"
            ] += 1
            counts[
                "incorrect_reuses"
            ] += 1
            counts[
                "false_semantic_hits"
            ] += 1

            if opportunity_exists:
                counts[
                    "missed_reuse_opportunities"
                ] += 1
            continue

        if label == "excluded_middle":
            counts[
                "excluded_middle_reuses"
            ] += 1
            counts[
                "excluded_middle_hits"
            ] += 1

            # Compatibility aliases.
            counts[
                "intermediate_reuses"
            ] += 1
            counts[
                "intermediate_semantic_hits"
            ] += 1

            if opportunity_exists:
                counts[
                    "missed_reuse_opportunities"
                ] += 1
            continue

        raise AssertionError(
            f"Unexpected RQ3 pair label: {label}"
        )

    # Complete RQ3 pair accounting guarantees this.
    if counts["unknown_selected_hits"] != 0:
        raise AssertionError(
            "unknown_selected_hits must remain zero with complete RQ3 ground truth"
        )

    if (
        int(counts["semantic_hits"])
        + int(counts["cache_misses"])
        != int(counts["total_queries"])
    ):
        raise AssertionError(
            "Cache accounting failure: hits + misses != total_queries"
        )

    if (
        int(counts["cache_misses"])
        != int(counts["cache_insertions"])
    ):
        raise AssertionError(
            "FIFO accounting failure: every miss must produce one insertion"
        )

    classified_hits = (
        int(counts["correct_semantic_hits"])
        + int(counts["false_semantic_hits"])
        + int(counts["excluded_middle_hits"])
    )

    if classified_hits != int(
        counts["semantic_hits"]
    ):
        raise AssertionError(
            "RQ3 hit accounting failure: "
            "correct + false + excluded_middle != semantic_hits"
        )

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
    Exact-query FIFO baseline.

    Because the RQ3 workload contains each textual query once, this should
    normally have zero hits.
    """
    cache: OrderedDict[int, None] = OrderedDict()
    counts = _new_counts()

    for query_idx in order:
        counts["total_queries"] += 1
        query = context.queries[query_idx]

        exact_idx: int | None = None

        for cached_idx in cache.keys():
            if (
                context.queries[cached_idx]
                == query
            ):
                exact_idx = cached_idx
                break

        if exact_idx is not None:
            counts["reuses"] += 1
            counts["semantic_hits"] += 1
            counts["scored_reuses"] += 1
            counts[
                "scored_semantic_hits"
            ] += 1
            counts["correct_reuses"] += 1
            counts[
                "correct_semantic_hits"
            ] += 1
            counts[
                "known_selected_reuses"
            ] += 1
            counts[
                "known_hit_jaccard_sum"
            ] += 1.0
            counts[
                "known_hit_jaccard_count"
            ] += 1
            counts[
                "reusable_opportunities"
            ] += 1

        else:
            counts["executions"] += 1
            counts["cache_misses"] += 1
            counts["cache_insertions"] += 1

            _insert_fifo(
                cache,
                query_idx,
                capacity,
            )

    metrics = _derive_metrics(counts)
    metrics["capacity"] = int(capacity)

    return metrics


# ============================================================
# Complete-RQ3 oracle baseline
# ============================================================


def _simulate_oracle_stream(
    context: OnlineCacheContext,
    order: list[int],
    capacity: int,
) -> dict[str, Any]:
    """
    Non-deployable oracle upper bound for the binary RQ3 task.

    Reuse iff at least one true RQ3-positive query is already in cache.
    Because RQ3 contains ALL positive test-query pairs, this opportunity
    accounting is complete rather than annotation-limited.
    """
    cache: OrderedDict[int, None] = OrderedDict()
    counts = _new_counts()

    for query_idx in order:
        counts["total_queries"] += 1

        cache_indices = list(
            cache.keys()
        )

        positives = (
            _positive_cached_candidates(
                context=context,
                query_idx=query_idx,
                cache_indices=cache_indices,
            )
        )

        if positives:
            counts[
                "reusable_opportunities"
            ] += 1
            counts["reuses"] += 1
            counts["semantic_hits"] += 1
            counts["scored_reuses"] += 1
            counts[
                "scored_semantic_hits"
            ] += 1
            counts["correct_reuses"] += 1
            counts[
                "correct_semantic_hits"
            ] += 1
            counts[
                "known_selected_reuses"
            ] += 1

            # All positive pairs have their exact J stored.
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

            if best_j is None:
                raise AssertionError(
                    "Positive RQ3 pair is missing its stored Jaccard value."
                )

            counts[
                "known_hit_jaccard_sum"
            ] += float(best_j)
            counts[
                "known_hit_jaccard_count"
            ] += 1

        else:
            counts["executions"] += 1
            counts["cache_misses"] += 1
            counts["cache_insertions"] += 1

            _insert_fifo(
                cache,
                query_idx,
                capacity,
            )

    metrics = _derive_metrics(counts)
    metrics["capacity"] = int(capacity)

    return metrics


# ============================================================
# Aggregation
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


def _aggregate_stream_metrics(
    runs: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    if not runs:
        return {}

    keys: set[str] = set()

    for run in runs:
        keys.update(run.keys())

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
            value = run.get(key)

            if isinstance(value, bool):
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
            "n": len(values),
        }

    return aggregated


# ============================================================
# Evaluate one capacity
# ============================================================


def _evaluate_capacity(
    context: OnlineCacheContext,
    stream_orders: list[
        tuple[int, list[int]]
    ],
    capacity: int,
    threshold: float,
    cache_cfg: dict[str, Any],
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

    for stream_seed, order in stream_orders:
        semantic = _simulate_semantic_stream(
            context=context,
            order=order,
            capacity=capacity,
            threshold=threshold,
        )

        semantic[
            "stream_seed"
        ] = int(stream_seed)

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
            ] = int(stream_seed)

            exact_runs.append(
                exact
            )

        if include_oracle:
            oracle = (
                _simulate_oracle_stream(
                    context=context,
                    order=order,
                    capacity=capacity,
                )
            )

            oracle[
                "stream_seed"
            ] = int(stream_seed)

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
# Threshold sensitivity
# ============================================================


def _evaluate_threshold_sensitivity(
    context: OnlineCacheContext,
    stream_orders: list[
        tuple[int, list[int]]
    ],
    capacity: int,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    """
    Robustness analysis only. Never selects the test threshold.
    """
    out: list[dict[str, Any]] = []

    for threshold in thresholds:
        runs: list[
            dict[str, Any]
        ] = []

        for stream_seed, order in stream_orders:
            metrics = (
                _simulate_semantic_stream(
                    context=context,
                    order=order,
                    capacity=capacity,
                    threshold=float(
                        threshold
                    ),
                )
            )

            metrics[
                "stream_seed"
            ] = int(stream_seed)

            runs.append(
                metrics
            )

        pooled = _pool_stream_counts(
            runs
        )

        pooled[
            "threshold"
        ] = float(threshold)

        out.append(
            pooled
        )

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
    Evaluate RQ3 using COMPLETE test-only pair ground truth.

    This function intentionally keeps the same signature as the existing
    experiment runner, so scripts/run_experiment.py does not need to change.

    validation_records is accepted for API compatibility but is NOT used for
    RQ3. The cosine threshold is fixed a priori, so validation is unnecessary
    for threshold selection.

    Runtime:
        1. embed incoming test query
        2. find nearest cached query by cosine similarity
        3. reuse iff cosine >= fixed threshold
        4. otherwise execute/miss and insert into FIFO

    Offline AFTER runtime choice:
        positive pair        -> correct semantic reuse
        negative pair        -> false semantic reuse
        absent from both     -> deliberately excluded middle pair

    Because the RQ3 builder evaluated every unordered test pair, there are no
    unannotated/unknown selected pairs.
    """
    del validation_records  # Explicitly unused in fixed-threshold RQ3.

    if not cache_cfg.get(
        "enabled",
        True,
    ):
        return {
            "definition": {
                "mode": "disabled"
            }
        }

    positive_jaccard_min = float(
        cache_cfg.get(
            "positive_jaccard_min",
            eval_cfg.get(
                "relevance_threshold",
                0.8,
            ),
        )
    )

    negative_jaccard_max = float(
        cache_cfg.get(
            "negative_jaccard_max",
            0.1,
        )
    )

    if (
        negative_jaccard_max
        >= positive_jaccard_min
    ):
        raise ValueError(
            "semantic_cache.negative_jaccard_max must be smaller than "
            "semantic_cache.positive_jaccard_min"
        )

    cosine_threshold = float(
        cache_cfg.get(
            "cosine_threshold",
            cache_cfg.get(
                "threshold",
                0.8,
            ),
        )
    )

    if not (
        -1.0
        <= cosine_threshold
        <= 1.0
    ):
        raise ValueError(
            "semantic_cache.cosine_threshold must lie in [-1, 1]"
        )

    encode_batch_size = int(
        cache_cfg.get(
            "encode_batch_size",
            eval_cfg.get(
                "encode_batch_size",
                64,
            ),
        )
    )

    capacities = [
        int(value)
        for value in cache_cfg.get(
            "capacities",
            [32, 64, 128, 256],
        )
    ]

    if (
        not capacities
        or any(
            capacity <= 0
            for capacity in capacities
        )
    ):
        raise ValueError(
            "semantic_cache.capacities must contain positive integers"
        )

    primary_capacity = int(
        cache_cfg.get(
            "primary_capacity",
            128,
        )
    )

    if primary_capacity <= 0:
        raise ValueError(
            "semantic_cache.primary_capacity must be positive"
        )

    if primary_capacity not in capacities:
        capacities = sorted(
            set(
                capacities
                + [primary_capacity]
            )
        )

    stream_seeds = [
        int(seed)
        for seed in cache_cfg.get(
            "stream_seeds",
            [100, 101, 102, 103, 104],
        )
    ]

    if not stream_seeds:
        raise ValueError(
            "semantic_cache.stream_seeds cannot be empty"
        )

    shuffle_streams = bool(
        cache_cfg.get(
            "shuffle_streams",
            True,
        )
    )

    # --------------------------------------------------------
    # Build TEST context only.
    # RQ3 path is automatically matched to the current test
    # query universe from the configured file(s).
    # --------------------------------------------------------

    test_context = _build_online_context(
        model=model,
        records=test_records,
        batch_size=encode_batch_size,
        cache_cfg=cache_cfg,
        positive_jaccard_min=positive_jaccard_min,
        negative_jaccard_max=negative_jaccard_max,
    )

    test_orders = [
        (
            seed,
            _make_stream_order(
                len(
                    test_context.queries
                ),
                seed,
                shuffle_streams,
            ),
        )
        for seed in stream_seeds
    ]

    capacity_results: dict[
        str,
        Any,
    ] = {}

    for capacity in capacities:
        capacity_results[
            str(capacity)
        ] = {
            "test": (
                _evaluate_capacity(
                    context=test_context,
                    stream_orders=test_orders,
                    capacity=capacity,
                    threshold=cosine_threshold,
                    cache_cfg=cache_cfg,
                )
            )
        }

    gt = test_context.ground_truth

    output: dict[
        str,
        Any,
    ] = {
        "definition": {
            "mode": (
                "offline fixed-threshold semantic-cache replay "
                "with complete test-only RQ3 pair ground truth"
            ),
            "workload_universe": (
                "all unique queries in the held-out test split: anchors plus "
                "positive and negative candidates"
            ),
            "workload_unique": True,
            "stream_order": (
                "seeded random permutation of the global unique test-query set"
                if shuffle_streams
                else "dataset-derived global test-query order"
            ),
            "cache_policy": "FIFO",
            "semantic_selection": (
                "nearest query currently in cache by cosine similarity"
            ),
            "runtime_reuse_rule": (
                f"semantic hit iff nearest-cache cosine >= "
                f"{cosine_threshold}"
            ),
            "threshold_mode": (
                "fixed a priori; no validation calibration"
            ),
            "cosine_threshold": float(
                cosine_threshold
            ),
            "ground_truth_source": (
                "complete test-only RQ3 pair file generated by executing every "
                "unique test query once and evaluating every unordered pair"
            ),
            "rq3_test_file": gt.path,
            "ground_truth_labels": {
                "positive": (
                    f"pair explicitly stored with J >= "
                    f"{positive_jaccard_min}"
                ),
                "negative": (
                    f"pair explicitly stored with J <= "
                    f"{negative_jaccard_max}"
                ),
                "excluded_middle": (
                    f"pair absent from both lists; complete pair accounting "
                    f"guarantees {negative_jaccard_max} < J < "
                    f"{positive_jaccard_min}"
                ),
                "unknown": (
                    "none: complete RQ3 pair accounting eliminates "
                    "unannotated pairs"
                ),
            },
            "runtime_state_rule": (
                "offline RQ3 labels never alter cache state: every "
                "cosine-accepted pair is reused and the incoming query is not "
                "inserted"
            ),
            "result_ground_truth_usage": (
                "RQ3 Jaccard-derived labels are used only after runtime "
                "selection for offline evaluation and by the non-deployable "
                "oracle baseline"
            ),
            "validation_usage": (
                "none for RQ3; threshold is fixed a priori"
            ),
            "primary_capacity": int(
                primary_capacity
            ),
            "capacities": capacities,
            "stream_seeds": stream_seeds,
            "baselines": {
                "exact_match": bool(
                    cache_cfg.get(
                        "include_exact_match_baseline",
                        True,
                    )
                ),
                "complete_rq3_oracle": bool(
                    cache_cfg.get(
                        "include_oracle_baseline",
                        True,
                    )
                ),
            },
        },

        # Kept for compatibility with existing aggregation code.
        "selected_threshold": float(
            cosine_threshold
        ),
        "fixed_threshold": float(
            cosine_threshold
        ),

        "validation": {
            "used_for_threshold_selection": False,
            "evaluated_for_rq3": False,
            "note": (
                "RQ3 uses test-only complete pair ground truth and a fixed "
                "a-priori cosine threshold; validation replay is intentionally "
                "not performed."
            ),
        },

        "capacities": (
            capacity_results
        ),

        "test_examples": len(
            test_context.queries
        ),

        "rq3_ground_truth": {
            "file": gt.path,
            "query_count": len(
                test_context.queries
            ),
            "all_unordered_pair_count": int(
                gt.all_pair_count
            ),
            "positive_pair_count": int(
                gt.positive_pair_count
            ),
            "negative_pair_count": int(
                gt.negative_pair_count
            ),
            "excluded_middle_pair_count": int(
                gt.excluded_middle_pair_count
            ),
            "unknown_pair_count": 0,
            "positive_jaccard_min": float(
                gt.positive_jaccard_min
            ),
            "negative_jaccard_max": float(
                gt.negative_jaccard_max
            ),
        },
    }

    # Optional robustness sweep. Never selects the headline threshold.
    if bool(
        cache_cfg.get(
            "run_threshold_sensitivity",
            False,
        )
    ):
        thresholds = [
            float(value)
            for value in cache_cfg.get(
                "thresholds",
                [
                    0.5,
                    0.6,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            )
        ]

        output[
            "threshold_sensitivity"
        ] = {
            "split": "test",
            "capacity": int(
                primary_capacity
            ),
            "note": (
                "robustness sweep only; headline threshold remains fixed "
                "a priori"
            ),
            "points": (
                _evaluate_threshold_sensitivity(
                    context=test_context,
                    stream_orders=test_orders,
                    capacity=primary_capacity,
                    thresholds=thresholds,
                )
            ),
        }

    return output
