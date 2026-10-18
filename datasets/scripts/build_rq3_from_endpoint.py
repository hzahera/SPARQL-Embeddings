#!/usr/bin/env python3
"""
Build test-only RQ3 semantic-cache ground truth directly from a SPARQL endpoint.

This version DOES NOT use or require query_execution_cache.jsonl.

For BOTH DBpedia and Wikidata it follows the same pipeline:

    test.json
        -> extract unique test queries
        -> execute every unique test query once
        -> keep answer sets only in memory
        -> compute Jaccard for every unordered query pair
        -> save only positive and negative pairs
        -> discard answer sets when the process exits

Endpoints
---------
DBpedia:
    https://dbpedia.data.dice-research.org/sparql

Wikidata:
    https://wikidata.data.dice-research.org/sparql

Pair labels
-----------
positive:
    Jaccard >= 0.8

negative:
    Jaccard <= 0.1

Pairs with:
    0.1 < Jaccard < 0.8

are evaluated but NOT stored in the final RQ3 JSON.

The final JSON does NOT contain answer sets.

Examples
--------

DBpedia:

python datasets/scripts/build_rq3_from_endpoint.py \
    --dataset dbpedia \
    --test datasets/dbpedia-splits/dbpedia-test.json \
    --output datasets/dbpedia-splits/dbpedia-test-rq3.json

Wikidata:

python datasets/scripts/build_rq3_from_endpoint.py \
    --dataset wikidata \
    --test datasets/wikidata-splits/wikidata-test.json \
    --output datasets/wikidata-splits/wikidata-test-rq3.json

Important
---------
This script intentionally keeps execution results only in memory.
If the process is interrupted during endpoint execution, it must restart the
query executions from the beginning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import requests
from tqdm import tqdm


DEFAULT_ENDPOINTS = {
    "dbpedia": "https://dbpedia.data.dice-research.org/sparql",
    "wikidata": "https://wikidata.data.dice-research.org/sparql",
}


# ---------------------------------------------------------------------------
# Query normalization / hashing
# Same normalization logic used in the clean-split builder.
# ---------------------------------------------------------------------------

def normalize_query(q: str) -> str:
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    return q


def query_hash(q: str) -> str:
    return hashlib.sha256(
        normalize_query(q).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Extract all unique queries from the TEST split only
# ---------------------------------------------------------------------------

def extract_unique_queries(test_path: Path) -> List[str]:
    with test_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected test JSON root to be a list: {test_path}"
        )

    by_norm: Dict[str, str] = {}

    def add(q: Any) -> None:
        if not isinstance(q, str) or not q.strip():
            return
        norm = normalize_query(q)
        by_norm.setdefault(norm, q.strip())

    for item in data:
        if not isinstance(item, dict):
            continue

        add(item.get("anchor_query"))

        for key in ("positive_queries", "negative_queries"):
            for candidate in item.get(key, []) or []:
                if isinstance(candidate, str):
                    add(candidate)
                elif isinstance(candidate, dict):
                    add(candidate.get("query"))

    # Deterministic ordering -> stable IDs across reruns.
    return [by_norm[k] for k in sorted(by_norm)]


# ---------------------------------------------------------------------------
# Result canonicalization
# Matches the existing clean-split construction logic.
# ---------------------------------------------------------------------------

def canonical_cell(
    binding: Dict[str, Any],
) -> Tuple[str, str, str, str]:
    return (
        str(binding.get("type", "")),
        str(binding.get("value", "")),
        str(binding.get("xml:lang", binding.get("lang", ""))),
        str(binding.get("datatype", "")),
    )


def canonical_row(
    binding: Dict[str, Any],
    vars_: Sequence[str],
) -> str:
    row = []

    for variable in vars_:
        if variable in binding:
            row.append(canonical_cell(binding[variable]))
        else:
            row.append(("", "", "", ""))

    return json.dumps(
        row,
        ensure_ascii=False,
        separators=(",", ":"),
    )


@dataclass
class ExecutionResult:
    query: str
    query_hash: str
    status: str
    answers: List[str]
    n_answers: int
    error: Optional[str] = None
    elapsed_s: Optional[float] = None


# ---------------------------------------------------------------------------
# Endpoint execution
# ---------------------------------------------------------------------------

def execute_one(
    query: str,
    endpoint: str,
    timeout_s: int,
    retries: int,
    user_agent: str,
) -> ExecutionResult:
    qh = query_hash(query)

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": user_agent,
    }

    last_error: Optional[str] = None

    for attempt in range(retries + 1):
        started = time.time()

        try:
            params = {
                "query": query,
                "format": "application/sparql-results+json",
            }

            response = requests.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=timeout_s,
            )

            # Retry long URL requests using POST.
            if response.status_code in (414, 431):
                response = requests.post(
                    endpoint,
                    data=params,
                    headers=headers,
                    timeout=timeout_s,
                )

            response.raise_for_status()
            payload = response.json()

            bindings = payload.get("results", {}).get("bindings")
            vars_ = payload.get("head", {}).get("vars", [])

            if bindings is None:
                return ExecutionResult(
                    query=query,
                    query_hash=qh,
                    status="unsupported_non_select",
                    answers=[],
                    n_answers=0,
                    error="Response does not contain results.bindings",
                    elapsed_s=time.time() - started,
                )

            answers = sorted(
                {
                    canonical_row(binding, vars_)
                    for binding in bindings
                }
            )

            return ExecutionResult(
                query=query,
                query_hash=qh,
                status="success" if answers else "empty",
                answers=answers,
                n_answers=len(answers),
                elapsed_s=time.time() - started,
            )

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))

    return ExecutionResult(
        query=query,
        query_hash=qh,
        status="error",
        answers=[],
        n_answers=0,
        error=last_error,
        elapsed_s=None,
    )


def execute_all_queries(
    queries: Sequence[str],
    endpoint: str,
    workers: int,
    timeout_s: int,
    retries: int,
    user_agent: str,
) -> List[ExecutionResult]:
    results: List[Optional[ExecutionResult]] = [
        None
    ] * len(queries)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_index = {
            pool.submit(
                execute_one,
                query,
                endpoint,
                timeout_s,
                retries,
                user_agent,
            ): idx
            for idx, query in enumerate(queries)
        }

        for future in tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc="Executing test queries",
        ):
            idx = future_to_index[future]
            results[idx] = future.result()

    final_results: List[ExecutionResult] = []

    for idx, result in enumerate(results):
        if result is None:
            raise RuntimeError(
                f"No execution result produced for query index {idx}."
            )
        final_results.append(result)

    return final_results


# ---------------------------------------------------------------------------
# Validate execution coverage
# ---------------------------------------------------------------------------

def validate_execution_results(
    queries: Sequence[str],
    results: Sequence[ExecutionResult],
) -> Tuple[List[Set[str]], List[dict]]:
    if len(queries) != len(results):
        raise RuntimeError(
            "Query/result length mismatch."
        )

    status_counts = Counter(r.status for r in results)

    print("\nExecution statuses:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:24s} {count:,}")

    failures: List[Tuple[int, ExecutionResult]] = []

    for idx, result in enumerate(results):
        if result.status != "success" or not result.answers:
            failures.append((idx, result))

    if failures:
        print("\n" + "=" * 80)
        print("RQ3 BUILD STOPPED: incomplete endpoint execution")
        print("=" * 80)
        print(
            f"{len(failures)} / {len(results)} test queries did not return "
            "a successful non-empty result."
        )

        print("\nFirst few failures:")

        for idx, result in failures[:10]:
            print("-" * 80)
            print(f"query_id : {idx}")
            print(f"status   : {result.status}")
            print(f"error    : {result.error}")
            print(f"query    : {result.query[:500]}")

        raise RuntimeError(
            "All RQ3 test queries must have successful non-empty results. "
            "Rerun after fixing endpoint/timeouts before generating the final "
            "RQ3 JSON."
        )

    answer_sets: List[Set[str]] = []
    query_rows: List[dict] = []

    for qid, result in enumerate(results):
        answers = set(result.answers)

        answer_sets.append(answers)

        query_rows.append(
            {
                "id": qid,
                "query_hash": result.query_hash,
                "query": result.query,
                "n_answers": len(answers),
            }
        )

    return answer_sets, query_rows


# ---------------------------------------------------------------------------
# Efficient exact pairwise Jaccard
# ---------------------------------------------------------------------------

def build_shared_answer_intersections(
    answer_sets: Sequence[Set[str]],
) -> Dict[Tuple[int, int], int]:
    """
    Construct an inverted index over canonical answer rows.

    Only query pairs sharing >=1 answer are stored in `intersections`.

    A pair absent from intersections has:
        intersection = 0
        Jaccard = 0

    because all accepted test queries are guaranteed non-empty.
    """
    inverted: Dict[str, List[int]] = defaultdict(list)

    for qid, answers in enumerate(
        tqdm(
            answer_sets,
            desc="Indexing answer sets",
        )
    ):
        for answer in answers:
            inverted[answer].append(qid)

    intersections: Dict[Tuple[int, int], int] = defaultdict(int)

    for query_ids in tqdm(
        inverted.values(),
        desc="Counting shared-answer intersections",
    ):
        if len(query_ids) < 2:
            continue

        ids = sorted(set(query_ids))

        for i, j in combinations(ids, 2):
            intersections[(i, j)] += 1

    return dict(intersections)


def jaccard_from_intersection(
    i: int,
    j: int,
    intersections: Dict[Tuple[int, int], int],
    sizes: Sequence[int],
) -> float:
    if i == j:
        return 1.0

    a, b = (i, j) if i < j else (j, i)

    intersection = intersections.get((a, b), 0)

    if intersection == 0:
        return 0.0

    union = sizes[a] + sizes[b] - intersection

    return (
        intersection / union
        if union
        else 1.0
    )


# ---------------------------------------------------------------------------
# RQ3 generation
# ---------------------------------------------------------------------------

def build_rq3(
    dataset: str,
    test_path: Path,
    output_path: Path,
    endpoint: str,
    workers: int,
    timeout_s: int,
    retries: int,
    user_agent: str,
    positive_min: float,
    negative_max: float,
) -> None:
    if not (
        0.0 <= negative_max < positive_min <= 1.0
    ):
        raise ValueError(
            "Expected 0 <= negative_max < positive_min <= 1."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("RQ3 TEST-ONLY GROUND-TRUTH GENERATOR")
    print("=" * 80)
    print(f"Dataset          : {dataset}")
    print(f"Test split       : {test_path}")
    print(f"Endpoint         : {endpoint}")
    print(f"Output           : {output_path}")
    print(f"Workers          : {workers}")
    print(f"Timeout          : {timeout_s}s")
    print(f"Retries          : {retries}")
    print(f"Positive         : J >= {positive_min}")
    print(f"Negative         : J <= {negative_max}")
    print(
        f"Excluded middle  : "
        f"{negative_max} < J < {positive_min}"
    )
    print("\nNo execution-cache JSONL will be read or written.")

    # ---------------------------------------------------------------
    # 1. Extract test-query universe
    # ---------------------------------------------------------------

    queries = extract_unique_queries(test_path)

    print(f"\nUnique test queries: {len(queries):,}")

    if len(queries) < 2:
        raise RuntimeError(
            "Need at least two unique test queries."
        )

    # ---------------------------------------------------------------
    # 2. Execute every test query directly on endpoint
    # ---------------------------------------------------------------

    print("\nExecuting every unique test query once...")

    results = execute_all_queries(
        queries=queries,
        endpoint=endpoint,
        workers=workers,
        timeout_s=timeout_s,
        retries=retries,
        user_agent=user_agent,
    )

    # ---------------------------------------------------------------
    # 3. Verify all query results before creating RQ3 file
    # ---------------------------------------------------------------

    answer_sets, query_rows = validate_execution_results(
        queries,
        results,
    )

    print(
        f"\nCoverage check: {len(answer_sets):,}/{len(queries):,} "
        "successful non-empty test queries."
    )

    # ---------------------------------------------------------------
    # 4. Compute all pair relationships
    # ---------------------------------------------------------------

    sizes = [len(answers) for answers in answer_sets]

    intersections = build_shared_answer_intersections(
        answer_sets
    )

    n = len(queries)
    total_pairs = n * (n - 1) // 2

    positive_pairs: List[List[Any]] = []
    negative_pairs: List[List[Any]] = []
    intermediate_count = 0

    print(
        f"\nClassifying all {total_pairs:,} unordered "
        "test-query pairs..."
    )

    for i in tqdm(
        range(n),
        desc="Classifying pairs",
    ):
        for j in range(i + 1, n):
            jac = jaccard_from_intersection(
                i=i,
                j=j,
                intersections=intersections,
                sizes=sizes,
            )

            # Classification is done using full precision.
            # Only serialized output is rounded.
            jac_out = round(
                float(jac),
                12,
            )

            if jac >= positive_min:
                positive_pairs.append(
                    [i, j, jac_out]
                )

            elif jac <= negative_max:
                negative_pairs.append(
                    [i, j, jac_out]
                )

            else:
                intermediate_count += 1

    accounted = (
        len(positive_pairs)
        + len(negative_pairs)
        + intermediate_count
    )

    if accounted != total_pairs:
        raise AssertionError(
            f"Pair accounting failed: "
            f"{accounted} != {total_pairs}"
        )

    # ---------------------------------------------------------------
    # 5. Final test-only RQ3 JSON
    # ---------------------------------------------------------------

    metadata = {
        "rq": "RQ3",
        "task": "semantic_cache_pair_ground_truth",
        "dataset": dataset,
        "split": "test",
        "endpoint": endpoint,
        "source_test_file": str(test_path),
        "query_count": n,
        "all_unordered_pair_count": total_pairs,
        "positive_jaccard_min": positive_min,
        "negative_jaccard_max": negative_max,
        "positive_pair_count": len(positive_pairs),
        "negative_pair_count": len(negative_pairs),
        "excluded_intermediate_pair_count": intermediate_count,
        "pair_coverage": (
            "complete over all unordered pairs of unique test queries"
        ),
        "omitted_pair_semantics": (
            f"A distinct test-query pair absent from both "
            f"positive_pairs and negative_pairs was evaluated and "
            f"satisfies {negative_max} < J < {positive_min}."
        ),
        "answer_sets_in_output": False,
        "execution_results_persisted": False,
        "runtime_ground_truth_rule": (
            "Jaccard is offline evaluation ground truth only. "
            "Semantic-cache runtime selection must use embedding cosine "
            "similarity only."
        ),
    }

    output = {
        "metadata": metadata,
        "queries": query_rows,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RQ3 BUILD COMPLETE")
    print("=" * 80)

    print(
        f"Queries                 : {n:,}"
    )
    print(
        f"All unordered pairs     : {total_pairs:,}"
    )
    print(
        f"Positive pairs          : "
        f"{len(positive_pairs):,}"
    )
    print(
        f"Negative pairs          : "
        f"{len(negative_pairs):,}"
    )
    print(
        f"Excluded intermediate   : "
        f"{intermediate_count:,}"
    )
    print(
        f"Pair accounting         : "
        f"{accounted:,}/{total_pairs:,}"
    )
    print(
        f"Answer sets saved       : NO"
    )
    print(
        f"Output size             : "
        f"{output_path.stat().st_size / (1024 ** 2):.2f} MiB"
    )
    print(
        f"Saved                   : {output_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute all unique test queries directly against the selected "
            "DICE endpoint and build complete RQ3 positive/negative pair "
            "ground truth without using an execution-cache JSONL."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=("dbpedia", "wikidata"),
    )

    parser.add_argument(
        "--test",
        required=True,
        type=Path,
        help="Existing clean test split JSON.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Final RQ3 JSON.",
    )

    parser.add_argument(
        "--endpoint",
        default=None,
        help=(
            "Optional endpoint override. By default the correct DICE endpoint "
            "is chosen from --dataset."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Parallel endpoint requests. Default: 4 for DBpedia, "
            "2 for Wikidata."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-query endpoint timeout in seconds. Default: 120.",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for failed requests. Default: 3.",
    )

    parser.add_argument(
        "--user-agent",
        default="result-aware-transformers-rq3/1.0",
    )

    parser.add_argument(
        "--positive-min",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--negative-max",
        type=float,
        default=0.1,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    endpoint = (
        args.endpoint
        if args.endpoint
        else DEFAULT_ENDPOINTS[args.dataset]
    )

    if args.workers is None:
        workers = (
            4
            if args.dataset == "dbpedia"
            else 2
        )
    else:
        workers = args.workers

    if workers < 1:
        raise ValueError(
            "--workers must be >= 1"
        )

    build_rq3(
        dataset=args.dataset,
        test_path=args.test,
        output_path=args.output,
        endpoint=endpoint,
        workers=workers,
        timeout_s=args.timeout,
        retries=args.retries,
        user_agent=args.user_agent,
        positive_min=args.positive_min,
        negative_max=args.negative_max,
    )


if __name__ == "__main__":
    main()
